import json
import os
import re
import time
from typing import Any, Generator, Mapping, Optional, Union

from dotenv import load_dotenv
from llamaapi import LlamaAPI  # type: ignore
from openai.types.chat import ChatCompletion

from ..core.core import Completion, PreferenceError, PreferenceResult, Ranker
from .build_llama_request import LLAMA_PAYOFF_LABELS, build_llama_request

load_dotenv()


class LlamaRanker(Ranker):
    """Run preference labelling using open-source models, such as Llama2.

    Uses deployed instance of [Llama](https://www.llama-api.com/) for inference.
    As of Nov 23, they offer a [variety of models](https://docs.llama-api.com/quickstart),
    including 'llama-70b-chat', 'mistral-7b-instruct' and 'falcon-40b-instruct'.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.llama = LlamaAPI(os.getenv("LLAMA_API_TOKEN"))

    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        request = build_llama_request(
            instruction, completion_a.completion, completion_b.completion, self.model
        )
        response = self._call_llama_api(request)
        if isinstance(response, PreferenceError):
            return response
        return self._to_result(response, completion_a, completion_b)

    def _call_llama_api(
        self, request: Mapping[str, Any], max_retries: int = 5
    ) -> Union[Any, PreferenceError]:
        retry_delays = self._exponential_backoff()
        error_trace = []
        for _ in range(max_retries):
            try:
                response = self.llama.run(request)
                return response.json()
            except Exception as e:
                error_trace.append(str(e))
                time.sleep(next(retry_delays))
        return PreferenceError(error_messages=error_trace)

    @staticmethod
    def _exponential_backoff(
        base_delay: float = 0.5, factor: float = 2, max_delay: float = 60
    ) -> Generator[float, None, None]:
        delay = base_delay
        while True:
            yield delay
            delay = min(delay * factor, max_delay)

    @staticmethod
    def _to_result(
        response: Any, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        try:
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError(f"Expected content to be string, got {type(content)}.")
            return PreferenceResult(
                completion_a=completion_a,
                completion_b=completion_b,
                payoff=LLAMA_PAYOFF_LABELS.payoff_from_string(content),
            )
        except Exception as e:
            return PreferenceError(error_messages=[str(e)], original_output=response)
