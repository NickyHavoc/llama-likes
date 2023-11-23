from pytest import fixture

from llama_likes import Completion, Model, OpenaiRanker, Payoff, PreferenceResult


@fixture
def openai_model() -> Model:
    return Model.GPT_3_5


@fixture
def openai_ranker(openai_model: Model) -> OpenaiRanker:
    return OpenaiRanker(openai_model)


def test_openai_ranker_true_positive(
    openai_ranker: OpenaiRanker, labeling_example: tuple[str, Completion, Completion]
) -> None:
    ranking = openai_ranker.rank(*labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
