from pytest import fixture

from llama_likes import (
    HuggingfaceModel,
    LlamaRanker,
    Payoff,
    PreferenceInput,
    PreferenceResult,
)


@fixture
def llama_model() -> HuggingfaceModel:
    return HuggingfaceModel.LLAMA_7B_CHAT


@fixture
def llama_ranker(llama_model: HuggingfaceModel) -> LlamaRanker:
    return LlamaRanker(llama_model)


def test_llama_ranker_true_positive(
    llama_ranker: LlamaRanker, labeling_example: PreferenceInput
) -> None:
    ranking = llama_ranker.rank(labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
