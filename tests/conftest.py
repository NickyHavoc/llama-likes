from pytest import fixture

from llama_likes import OpenAiRanker


@fixture
def openai_model() -> str:
    return "gpt-3.5-turbo-1106"


@fixture
def openai_ranker(openai_model: str) -> OpenAiRanker:
    return OpenAiRanker(openai_model)
