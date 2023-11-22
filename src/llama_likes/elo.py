from typing import Mapping, Sequence

from .core.core import Payoff, PreferenceResult


class Elo:
    def __init__(self, players: Sequence[str], k_factor: int = 20) -> None:
        self.ratings: dict[str, float] = {
            p: 1500 for p in players
        }
        self._k_factor = k_factor

    def calculate_batch(self, preference_results: Sequence[PreferenceResult]) -> None:
        for result in preference_results:
            self.calculate(result)

    def calculate(self, preference_result: PreferenceResult) -> None:
        player_a = preference_result.completion_a.player_id
        player_b = preference_result.completion_b.player_id

        expected_win_rate_a, expected_win_rate_b = self._calculate_expected_win_rates(player_a, player_b)

        self._update_ratings(player_a, player_b, preference_result.payoff, expected_win_rate_a, expected_win_rate_b)

    def _get_rating(self, player_id: str) -> float:
        score = self.ratings.get(player_id)
        if not score:
            raise ValueError(f"Got unkown player: {player_id}")
        return score

    def _calculate_expected_win_rates(self, player_a: str, player_b: str) -> tuple[float, float]:
        rating_a, rating_b = self._get_rating(player_a), self._get_rating(player_b)
        expected_win_rate_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected_win_rate_a, 1 - expected_win_rate_a

    def _update_ratings(self, player_a: str, player_b: str, payoff: Payoff, expected_win_rate_a: float, expected_win_rate_b: float) -> None:
        actual_a, actual_b = payoff.value
        self._calc_and_set_new_rating(player_a, actual_a, expected_win_rate_a)
        self._calc_and_set_new_rating(player_b, actual_b, expected_win_rate_b)

    def _calc_and_set_new_rating(self, player_id: str, actual: float, expected_win_rate: float) -> None:
        current_rating = self._get_rating(player_id)
        new_rating = current_rating + self._k_factor * (actual - expected_win_rate)
        self.ratings[player_id] = new_rating
