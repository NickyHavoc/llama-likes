from typing import Sequence
from pytest import fixture

from llama_likes import Completion, Payoff, PreferenceResult
from llama_likes import Elo


PLAYER_A_NAME = "p1"
PLAYER_B_NAME = "p2"


@fixture
def elo() -> Elo:
    return Elo(players=[PLAYER_A_NAME, PLAYER_B_NAME])


@fixture
def preference_results() -> Sequence[PreferenceResult]:
    return [
        PreferenceResult(
            completion_a=Completion(
                player_id=PLAYER_A_NAME,
                completion="Good answer."
            ),
            completion_b=Completion(
                player_id=PLAYER_B_NAME,
                completion="Bad answer."
            ),
            payoff=Payoff.PLAYER_A_WINS
        ),
        PreferenceResult(
            completion_a=Completion(
                player_id=PLAYER_A_NAME,
                completion="Mediocre answer."
            ),
            completion_b=Completion(
                player_id=PLAYER_B_NAME,
                completion="Mediocre answer."
            ),
            payoff=Payoff.DRAW
        ),
    ]


def test_elo_ranks_better_player_higher(elo: Elo, preference_results: Sequence[PreferenceResult]) -> None:
    elo.calculate_batch(preference_results)

    assert elo._get_rating(PLAYER_A_NAME) > elo._get_rating(PLAYER_B_NAME)
