import json
import time
from typing import Generator, Mapping, Sequence, Union

import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

from llama_likes.core import Completion, PreferenceError, PreferenceResult

from .build_openai_prompt import OPENAI_PAYOFF_LABELS, build_openai_prompt

load_dotenv()


class OpenAiRanker:
    def __init__(self, model: str) -> None:
        self.model = model

    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        prompt = build_openai_prompt(
            instruction, completion_a.completion, completion_b.completion
        )
        response = self._call_openai_api(prompt)
        if isinstance(response, PreferenceError):
            return response
        return self._to_result(response, completion_a, completion_b)

    def _call_openai_api(
        self, messages: Sequence[Mapping[str, str]], max_retries: int = 5
    ) -> Union[ChatCompletion, PreferenceError]:
        retry_delays = self._exponential_backoff()
        error_trace = []
        for _ in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=5,
                )  # type: ignore
                assert isinstance(response, ChatCompletion)
                return response
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
        response: ChatCompletion, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError(f"Expected content to be string, got {type(content)}.")
            loaded = json.loads(content)
            return PreferenceResult(
                completion_a=completion_a,
                completion_b=completion_b,
                payoff=OPENAI_PAYOFF_LABELS.get_payoff(loaded),
            )
        except Exception as e:
            return PreferenceError(
                error_messages=[str(e)], original_output=response.model_dump()
            )
