from pytest import fixture

from llama_likes import Completion, OpenAiRanker, Payoff, PreferenceResult


@fixture
def example() -> tuple[str, Completion, Completion]:
    return (
        "What is 1 + 1?",
        Completion(player_id="p1", completion="2"),
        Completion(player_id="p2", completion="1"),
    )


def test_ranker_true_positive(
    openai_ranker: OpenAiRanker, example: tuple[str, Completion, Completion]
) -> None:
    ranking = openai_ranker.rank(*example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
