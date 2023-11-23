from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Union

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

    def _labels(self) -> Sequence[str]:
        return [self.player_a_wins, self.player_a_wins, self.draw]

    def _payoff_map(self) -> Mapping[str, Payoff]:
        return {
            self.player_a_wins: Payoff.PLAYER_A_WINS,
            self.draw: Payoff.DRAW,
            self.player_b_wins: Payoff.PLAYER_B_WINS,
        }

    def payoff_from_json(self, generated_json: Mapping[str, str]) -> Payoff:
        payoff_key = generated_json[self.json_key]
        return self._payoff_map()[payoff_key]

    def payoff_from_string(self, string: str) -> Payoff:
        payoff_key = next(label for label in self._labels() if label in string)
        return self._payoff_map()[payoff_key]


class PreferenceError(BaseModel):
    error_messages: Sequence[str]
    original_output: Optional[Any] = None


class PreferenceResult(BaseModel):
    completion_a: Completion
    completion_b: Completion
    payoff: Payoff


class Ranker(ABC):
    @abstractmethod
    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        ...
