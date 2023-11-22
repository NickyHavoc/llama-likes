from typing import Mapping, Sequence

from llama_likes.core.core import PayoffLabels

OPENAI_PAYOFF_LABELS = PayoffLabels(
    player_a_wins="completion_a",
    draw="equal",
    player_b_wins="completion_b",
    json_key="which_completion_is_better",
)

_system_message = {
    "role": "system",
    "content": f"""Find an instruction and two completions below. Then, proceed to rank which completion is better. If they are equally good or bad, return "{OPENAI_PAYOFF_LABELS.draw}".
Consider the following points when ranking the completions:
- Correctness
- Adherence of the completion to the instruction
- Creativity (if desired)

If a completion ends abruptly (mid-sentence), consider how good it was until it ended. Do not deduct points for this!
Make sure that your preferences are logical. For example, if the completions are equally good, they must get the same score!

Output the following json format:
```
{{
    "{OPENAI_PAYOFF_LABELS.json_key}": Literal["{OPENAI_PAYOFF_LABELS.player_a_wins}", "{OPENAI_PAYOFF_LABELS.player_b_wins}", "{OPENAI_PAYOFF_LABELS.draw}"]
}}
```""",
}
_user_template = """instruction:
{instruction}

completion_a:
{completion_a}

completion_b:
{completion_b}"""


def build_openai_prompt(
    instruction: str, completion_a: str, completion_b: str
) -> Sequence[Mapping[str, str]]:
    return [
        _system_message,
        {
            "role": "user",
            "content": _user_template.format(
                instruction=instruction,
                completion_a=completion_a,
                completion_b=completion_b,
            ),
        },
    ]
