from typing import Union

from ..core.core import (
    Completion,
    HuggingfaceModel,
    PreferenceError,
    PreferenceInput,
    PreferenceResult,
    Ranker,
)
from ..core.huggingface_client import HuggingfaceClient, HuggingfaceCompletionRequest
from .build_llama_request import LLAMA_PAYOFF_LABELS, build_prompt


class LlamaRanker(Ranker):
    """Run preference labelling using open-source models, such as Llama2.

    Uses deployed instance of [Llama](https://www.llama-api.com/) for inference.
    As of Nov 23, they offer a [variety of models](https://docs.llama-api.com/quickstart),
    including 'llama-70b-chat', 'mistral-7b-instruct' and 'falcon-40b-instruct'.
    """

    def __init__(self, model: HuggingfaceModel) -> None:
        self.huggingface_client = HuggingfaceClient(model)

    def rank(
        self, preference_input: PreferenceInput
    ) -> Union[PreferenceResult, PreferenceError]:
        prompt = build_prompt(
            preference_input.instruction,
            preference_input.completion_a.completion,
            preference_input.completion_b.completion,
            self.huggingface_client,
        )
        request = self._build_request(prompt)
        response = self._call_api(request)
        if isinstance(response, PreferenceError):
            return response
        return self._to_result(response, preference_input)

    @staticmethod
    def _build_request(prompt: str) -> HuggingfaceCompletionRequest:
        return HuggingfaceCompletionRequest(
            inputs=prompt, temperature=0.01, max_new_tokens=24
        )  # temperature must be > 0

    @Ranker.retry_with_backoff()
    def _call_api(self, request: HuggingfaceCompletionRequest) -> str:
        return self.huggingface_client.complete(request)

    @staticmethod
    def _to_result(
        response: str, preference_input: PreferenceInput
    ) -> Union[PreferenceResult, PreferenceError]:
        try:
            return PreferenceResult(
                preference_input=preference_input,
                payoff=LLAMA_PAYOFF_LABELS.payoff_from_string(response),
            )
        except Exception as e:
            return PreferenceError(error_messages=[str(e)], original_output=response)
