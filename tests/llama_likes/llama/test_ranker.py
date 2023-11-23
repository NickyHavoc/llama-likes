from typing import Union
import pytest
from llama_likes import Ranker, Completion, PreferenceResult, PreferenceError


MAX_RETRIES = 3


class DummyRanker(Ranker):
    def rank(
        self, instruction: str, completion_a: Completion, completion_b: Completion
    ) -> Union[PreferenceResult, PreferenceError]:
        return PreferenceError()

    @Ranker.retry_with_backoff(max_retries=MAX_RETRIES, base_delay=0.01, max_delay=1)
    def fails_and_catches_error(self) -> None:
        raise TimeoutError("Fake Timeout")
    
    @Ranker.retry_with_backoff()
    def fails_and_raises(self) -> None:
        raise BaseException()


@pytest.fixture
def ranker() -> DummyRanker:
    return DummyRanker()


def test_ranker_catches_timeout_errors(ranker: DummyRanker) -> None:
    output = ranker.fails_and_catches_error()
    assert isinstance(output, PreferenceError)
    assert len(output.error_messages) == MAX_RETRIES


def test_ranker_raises_on_other_exception(ranker: DummyRanker) -> None:
    with pytest.raises(BaseException):
        ranker.fails_and_raises()
