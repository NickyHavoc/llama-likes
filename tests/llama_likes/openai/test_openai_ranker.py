from llama_likes import Completion, OpenaiRanker, Payoff, PreferenceResult


def test_openai_ranker_true_positive(
    openai_ranker: OpenaiRanker, labeling_example: tuple[str, Completion, Completion]
) -> None:
    ranking = openai_ranker.rank(*labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
