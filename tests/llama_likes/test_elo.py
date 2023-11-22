from typing import Sequence

from pytest import fixture

from llama_likes import Completion, Elo, Payoff, PreferenceResult
from llama_likes.elo import elo_simulation

PLAYER_A_NAME = "p1"
PLAYER_B_NAME = "p2"


@fixture
def preference_results() -> Sequence[PreferenceResult]:
    return [
        PreferenceResult(
            completion_a=Completion(player_id=PLAYER_A_NAME, completion="Good answer."),
            completion_b=Completion(player_id=PLAYER_B_NAME, completion="Bad answer."),
            payoff=Payoff.PLAYER_A_WINS,
        ),
        PreferenceResult(
            completion_a=Completion(player_id=PLAYER_A_NAME, completion="Ok answer."),
            completion_b=Completion(player_id=PLAYER_B_NAME, completion="Ok answer."),
            payoff=Payoff.DRAW,
        ),
        PreferenceResult(
            completion_a=Completion(player_id=PLAYER_A_NAME, completion="Bad answer."),
            completion_b=Completion(player_id=PLAYER_B_NAME, completion="Good answer."),
            payoff=Payoff.PLAYER_B_WINS,
        ),
        PreferenceResult(
            completion_a=Completion(player_id=PLAYER_A_NAME, completion="Good answer."),
            completion_b=Completion(player_id=PLAYER_B_NAME, completion="Bad answer."),
            payoff=Payoff.PLAYER_A_WINS,
        ),
    ]


def test_elo_ranks_better_player_higher(
    preference_results: Sequence[PreferenceResult],
) -> None:
    elo = Elo(
        players=[PLAYER_A_NAME, PLAYER_B_NAME]
    )  # not using fixture as class is stateful
    elo.calculate_batch(preference_results)

    assert elo._get_rating(PLAYER_A_NAME) > elo._get_rating(PLAYER_B_NAME)


def test_elo_simulation_yields_different_results(
    preference_results: Sequence[PreferenceResult],
) -> None:
    elo = Elo(players=[PLAYER_A_NAME, PLAYER_B_NAME])
    elo.calculate_batch(preference_results)
    single_run_result = elo.ratings

    simulation_result = elo_simulation(20, preference_results)

    assert single_run_result.get(PLAYER_A_NAME) != simulation_result.get(PLAYER_A_NAME)
    assert single_run_result.get(PLAYER_B_NAME) != simulation_result.get(PLAYER_B_NAME)
