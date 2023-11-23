from pytest import fixture

from llama_likes import Completion


@fixture
def labeling_example() -> tuple[str, Completion, Completion]:
    return (
        "What is 1 + 1?",
        Completion(player_id="p1", completion="2"),
        Completion(player_id="p2", completion="1"),
    )
