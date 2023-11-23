import json
import time
from typing import Generator, Mapping, Sequence, Union

import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

from ..core.core import Completion, Model, PreferenceError, PreferenceResult, Ranker
from .build_openai_prompt import OPENAI_PAYOFF_LABELS, build_prompt

load_dotenv()


class OpenaiRanker(Ranker):
    def __init__(self, model: Model) -> None:
        self.model = model.value

    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        prompt = build_prompt(
            instruction, completion_a.completion, completion_b.completion
        )
        response = self._call_api(prompt)
        if isinstance(response, PreferenceError):
            return response
        return self._to_result(response, completion_a, completion_b)

    @Ranker.retry_with_backoff()
    def _call_api(
        self, messages: Sequence[Mapping[str, str]], max_retries: int = 5
    ) -> Union[ChatCompletion, PreferenceError]:
        response = openai.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
            timeout=5,
        )  # type: ignore
        return response if isinstance(response, ChatCompletion) else PreferenceError()

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
                payoff=OPENAI_PAYOFF_LABELS.payoff_from_json(loaded),
            )
        except Exception as e:
            return PreferenceError(
                error_messages=[str(e)], original_output=response.model_dump()
            )
