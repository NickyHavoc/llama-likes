from typing import Mapping, Sequence


_starting_message = {
    "role": "system",
    "content": """Find an instruction and two completions below. Then, proceed to rank which completion is better. If they are equally good or bad, return "equal".
Next, grade the quality of each completion on a scale from 1 to 3, where 1 means "bad", 2 means "ok" and 3 means "good".
Consider the following points when grading and ranking the completions:
- Correctness 
- Adherence of the completion to the instruction
- Creativity (if desired)

If a completion ends abruptly (mid-sentence), consider how good it was until it ended. Do not deduct points for this!
Make sure that your preferences are logical. For example, if the completions are equally good, they must get the same score!

Output the following json format:
```
{
    "which_completion_is_better": Literal["completion_a", "completion_b", "equal"],
    "quality_completion_a": Literal[1, 2, 3],
    "quality_completion_b": Literal[1, 2, 3],
}
```"""
}
_input_template = """instruction:
{instruction}

completion_a:
{completion_a}

completion_b:
{completion_b}"""

def build_openai_prompt(instruction: str, completion_a: str, completion_b: str) -> Sequence[Mapping[str, str]]:
    return [
        _starting_message,
        {
            "role": "user",
            "content": _input_template.format(instruction=instruction, completion_a=completion_a, completion_b=completion_b)
        }
    ]
