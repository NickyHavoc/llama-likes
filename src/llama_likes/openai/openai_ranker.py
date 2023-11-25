import json
import time
from typing import Generator, Mapping, Sequence, Union

import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

from ..core.core import (
    Completion,
    OpenaiModel,
    PreferenceError,
    PreferenceInput,
    PreferenceResult,
    Ranker,
)
from .build_openai_prompt import OPENAI_PAYOFF_LABELS, build_prompt

load_dotenv()


class OpenaiRanker(Ranker):
    def __init__(self, model: OpenaiModel) -> None:
        self.model = model.value

    def rank(
        self, preference_input: PreferenceInput
    ) -> Union[PreferenceResult, PreferenceError]:
        prompt = build_prompt(
            preference_input.instruction,
            preference_input.completion_a.completion,
            preference_input.completion_b.completion,
        )
        response = self._call_api(prompt)
        if isinstance(response, PreferenceError):
            return response
        return self._to_result(response, preference_input)

    @Ranker.retry_with_backoff()
    def _call_api(
        self, messages: Sequence[Mapping[str, str]]
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
        response: ChatCompletion, preference_input: PreferenceInput
    ) -> Union[PreferenceResult, PreferenceError]:
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError(f"Expected content to be string, got {type(content)}.")
            loaded = json.loads(content)
            return PreferenceResult(
                preference_input=preference_input,
                payoff=OPENAI_PAYOFF_LABELS.payoff_from_json(loaded),
            )
        except Exception as e:
            return PreferenceError(
                error_messages=[str(e)], original_output=response.model_dump()
            )
