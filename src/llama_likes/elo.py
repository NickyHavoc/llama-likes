import random
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .core.core import Payoff, PreferenceResult


def elo_simulation(
    n: int, preference_results: Sequence[PreferenceResult]
) -> Mapping[str, float]:
    """
    Running just one elo yields unpredictable results, as the calculation
    is highly dependent on the order of results.
    """

    def get_players() -> Iterable[str]:
        return set(
            p
            for r in preference_results
            for p in [
                r.preference_input.completion_a.player_id,
                r.preference_input.completion_b.player_id,
            ]
        )

    players = get_players()
    ratings: dict[str, list[float]] = {player: [] for player in players}

    for index in range(n):
        random.seed(index)
        shuffled_results = list(preference_results)
        random.shuffle(shuffled_results)
        elo = Elo(players)
        elo.calculate_batch(shuffled_results)
        for player, rating in elo.ratings.items():
            ratings[player].append(rating)

    return {p: mean(r) for p, r in ratings.items()}


class Elo:
    def __init__(self, players: Iterable[str], k_factor: int = 20) -> None:
        self.ratings: dict[str, float] = {p: 1500 for p in players}
        self._k_factor = k_factor

    def calculate_batch(self, preference_results: Sequence[PreferenceResult]) -> None:
        for result in preference_results:
            self.calculate(result)

    def calculate(self, preference_result: PreferenceResult) -> None:
        player_a = preference_result.preference_input.completion_a.player_id
        player_b = preference_result.preference_input.completion_b.player_id

        expected_win_rate_a, expected_win_rate_b = self._calculate_expected_win_rates(
            player_a, player_b
        )

        self._update_ratings(
            player_a,
            player_b,
            preference_result.payoff,
            expected_win_rate_a,
            expected_win_rate_b,
        )

    def _get_rating(self, player_id: str) -> float:
        score = self.ratings.get(player_id)
        if not score:
            raise ValueError(f"Got unknown player: {player_id}")
        return score

    def _calculate_expected_win_rates(
        self, player_a: str, player_b: str
    ) -> tuple[float, float]:
        rating_a, rating_b = self._get_rating(player_a), self._get_rating(player_b)
        expected_win_rate_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected_win_rate_a, 1 - expected_win_rate_a

    def _update_ratings(
        self,
        player_a: str,
        player_b: str,
        payoff: Payoff,
        expected_win_rate_a: float,
        expected_win_rate_b: float,
    ) -> None:
        actual_a, actual_b = payoff.value
        self._calc_and_set_new_rating(player_a, actual_a, expected_win_rate_a)
        self._calc_and_set_new_rating(player_b, actual_b, expected_win_rate_b)

    def _calc_and_set_new_rating(
        self, player_id: str, actual: float, expected_win_rate: float
    ) -> None:
        current_rating = self._get_rating(player_id)
        new_rating = current_rating + self._k_factor * (actual - expected_win_rate)
        self.ratings[player_id] = new_rating
