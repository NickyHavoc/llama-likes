from typing import Any, Mapping

from llama_likes.core.core import PayoffLabels

# the llama2 instance used here is hosted by somebody somewhere
# docs are terribly & function calling doesn't work
# looking to change this, once a better alternative is available

LLAMA_PAYOFF_LABELS = PayoffLabels(
    player_a_wins="completion_a",
    draw="equal",
    player_b_wins="completion_b",
    json_key="",
)

_system_message = {
    "role": "system",
    "content": f"""Find an instruction and two completions below. Then, proceed to rank which completion is better. If they are equally good or bad, return "{LLAMA_PAYOFF_LABELS.draw}".
Consider the following points when ranking the completions:
- Correctness
- Adherence of the completion to the instruction
- Creativity (if desired)

If a completion ends abruptly (mid-sentence), consider how good it was until it ended. Do not deduct points for this!

Respond ONLY with "{LLAMA_PAYOFF_LABELS.player_a_wins}", "{LLAMA_PAYOFF_LABELS.player_b_wins}" or "{LLAMA_PAYOFF_LABELS.draw}".""",
}

_user_template = """instruction:
{instruction}

completion_a:
{completion_a}

completion_b:
{completion_b}"""


def build_llama_request(
    instruction: str, completion_a: str, completion_b: str, model: str
) -> Mapping[str, Any]:
    return {
        "messages": [
            _system_message,
            {
                "role": "user",
                "content": _user_template.format(
                    instruction=instruction,
                    completion_a=completion_a,
                    completion_b=completion_b,
                ),
            },
        ],
        "model": model
        # these parameters, while they would be great, appear to have no effect
        # API is terribly documented, no idea which parameters are available
        # "temperature": 0,
        # "max_length": 50,
        # "stop": "}",
    }
