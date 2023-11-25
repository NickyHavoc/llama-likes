from pytest import fixture

from llama_likes import Completion, PreferenceInput


@fixture
def labeling_example() -> PreferenceInput:
    return PreferenceInput(
        instruction="What is 1 + 1?",
        completion_a=Completion(player_id="p1", completion="2"),
        completion_b=Completion(player_id="p2", completion="1"),
    )
