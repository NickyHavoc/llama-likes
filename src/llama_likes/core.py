from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel


class Completion(BaseModel):
    player_id: str
    completion: str


class Payoff(Enum):
    PLAYER_A_WINS = (1, 0)
    DRAW = (0.5, 0.5)
    PLAYER_B_WINS = (0, 1)


class PayoffLabels(BaseModel):
    player_a_wins: str
    draw: str
    player_b_wins: str
    json_key: str

    def _payoff_map(self) -> Mapping[str, Payoff]:
        return {
            self.player_a_wins: Payoff.PLAYER_A_WINS,
            self.draw: Payoff.DRAW,
            self.player_b_wins: Payoff.PLAYER_B_WINS,
        }

    def get_payoff(self, generated_json: Mapping[str, str]) -> Payoff:
        payoff_key = generated_json[self.json_key]
        return self._payoff_map()[payoff_key]


class PreferenceError(BaseModel):
    error_messages: Sequence[str]
    original_output: Optional[Any] = None


class PreferenceResult(BaseModel):
    completion_a: Completion
    completion_b: Completion
    payoff: Payoff
