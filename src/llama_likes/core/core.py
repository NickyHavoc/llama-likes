import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generator, Mapping, Optional, Sequence, TypeVar, Union

from pydantic import BaseModel


class Model(Enum):
    """Model enum to list available models and the path to them."""

    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.1"
    LLAMA_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"
    GPT_3_5 = "gpt-3.5-turbo-1106"
    GPT_4 = "gpt-4-1106-preview"


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
    error_messages: Sequence[str] = []
    original_output: Optional[Any] = None


class PreferenceResult(BaseModel):
    completion_a: Completion
    completion_b: Completion
    payoff: Payoff


R = TypeVar("R")


class Ranker(ABC):
    @abstractmethod
    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        ...

    @staticmethod
    def retry_with_backoff(
        max_retries: int = 5,
        base_delay: float = 0.5,
        factor: float = 2,
        max_delay: float = 60,
    ) -> Callable[
        [Callable[..., Union[R, PreferenceError]]],
        Callable[..., Union[R, PreferenceError]],
    ]:
        """Not sure if this is good or terrible."""

        def decorator(
            func: Callable[..., Union[R, PreferenceError]]
        ) -> Callable[..., Union[R, PreferenceError]]:
            @wraps(func)
            def wrapper(
                *args: tuple[Any, ...], **kwargs: Mapping[str, Any]
            ) -> Union[R, PreferenceError]:
                def exponential_backoff() -> Generator[float, None, None]:
                    delay = base_delay
                    while True:
                        yield delay
                        delay = min(delay * factor, max_delay)

                retry_delays = exponential_backoff()
                error_trace = []

                for _ in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except TimeoutError as e:
                        error_trace.append(str(e))
                        time.sleep(next(retry_delays))
                    except Exception as e:
                        error_trace.append(str(e))
                        raise e
                return PreferenceError(error_messages=error_trace)

            return wrapper

        return decorator
