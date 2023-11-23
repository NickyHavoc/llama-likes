from pytest import fixture

from llama_likes import OpenaiRanker
from llama_likes.core.core import Completion
from llama_likes.llama.llama_ranker import LlamaRanker


@fixture
def openai_model() -> str:
    return "gpt-3.5-turbo-1106"


@fixture
def openai_ranker(openai_model: str) -> OpenaiRanker:
    return OpenaiRanker(openai_model)


@fixture
def llama_model() -> str:
    return "llama-70b-chat"


@fixture
def llama_ranker(llama_model: str) -> LlamaRanker:
    return LlamaRanker(llama_model)


@fixture
def labeling_example() -> tuple[str, Completion, Completion]:
    return (
        "What is 1 + 1?",
        Completion(player_id="p1", completion="2"),
        Completion(player_id="p2", completion="1"),
    )
