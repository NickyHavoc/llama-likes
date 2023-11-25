from typing import Mapping, Sequence

from pytest import fixture

from llama_likes import HuggingfaceClient, HuggingfaceModel
from llama_likes.core.huggingface_client import HuggingfaceCompletionRequest


@fixture
def mistral_model() -> HuggingfaceModel:
    return HuggingfaceModel.MISTRAL_7B_INSTRUCT


@fixture
def huggingface_client(mistral_model: HuggingfaceModel) -> HuggingfaceClient:
    return HuggingfaceClient(mistral_model)


@fixture
def messages() -> Sequence[Mapping[str, str]]:
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi, how are you?"},
        {"role": "user", "content": "Good & you?"},
    ]


def test_huggingface_client_wraps_chat_prompt(
    huggingface_client: HuggingfaceClient, messages: Sequence[Mapping[str, str]]
) -> None:
    prompt = huggingface_client.messages_to_prompt(messages)
    original_text = "".join(m["content"] for m in messages)

    assert isinstance(prompt, str)
    assert len(prompt) > len(original_text)


def test_huggingface_client_completes(
    huggingface_client: HuggingfaceClient, messages: Sequence[Mapping[str, str]]
) -> None:
    request = HuggingfaceCompletionRequest(
        inputs=huggingface_client.messages_to_prompt(messages),
        max_new_tokens=16,
        temperature=0.01,
    )
    response = huggingface_client.complete(request)

    assert any(word in response.lower() for word in ["good", "thank", "feel", "I'm"])
