from llama_likes import Completion, LlamaRanker, Payoff, PreferenceResult


def test_llama_ranker_true_positive(
    llama_ranker: LlamaRanker, labeling_example: tuple[str, Completion, Completion]
) -> None:
    ranking = llama_ranker.rank(*labeling_example)

    assert isinstance(ranking, PreferenceResult)
    assert ranking.payoff == Payoff.PLAYER_A_WINS
