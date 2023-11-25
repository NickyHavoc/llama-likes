from pytest import fixture

from llama_likes import (
    Completion,
    HuggingfaceModel,
    LlamaRanker,
    Payoff,
    PreferenceResult,
)


@fixture
def llama_model() -> HuggingfaceModel:
    return HuggingfaceModel.LLAMA_13B_CHAT


@fixture
def llama_ranker(llama_model: HuggingfaceModel) -> LlamaRanker:
    return LlamaRanker(llama_model)


def test_llama_ranker_true_positive(
    llama_ranker: LlamaRanker, labeling_example: tuple[str, Completion, Completion]
) -> None:
    ranking = llama_ranker.rank(*labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
