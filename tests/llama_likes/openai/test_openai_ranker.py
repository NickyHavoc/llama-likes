from pytest import fixture

from llama_likes import (
    OpenaiModel,
    OpenaiRanker,
    Payoff,
    PreferenceInput,
    PreferenceResult,
)


@fixture
def openai_model() -> OpenaiModel:
    return OpenaiModel.GPT_3_5


@fixture
def openai_ranker(openai_model: OpenaiModel) -> OpenaiRanker:
    return OpenaiRanker(openai_model)


def test_openai_ranker_true_positive(
    openai_ranker: OpenaiRanker, labeling_example: PreferenceInput
) -> None:
    ranking = openai_ranker.rank(labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
