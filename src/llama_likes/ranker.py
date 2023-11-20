import time
from typing import Literal, Mapping, Optional, Sequence, Union

from dotenv import load_dotenv
import openai
from openai import OpenAI
from pydantic import BaseModel 

load_dotenv()


class GradeResult(BaseModel):
    preference_label: Literal[1, 2, 3]
    quality_completion_a: Literal[1, 2, 3]
    quality_completion_b: Literal[1, 2, 3]


class GradeError(BaseModel):
    error_message: str
    original_output: Optional[str]


class Ranker:
    def __init__(self, model: str) -> None:
        self.model = model
        # self.client = OpenAI() # assumes you have OPENAI_API_KEY set

    def rank_openai(self, instruction: str, completion_a: str, completion_b: str) -> Union[GradeResult, GradeError]:
        response = None
        retry_count = 0
        while not response and retry_count < 5:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=5
                )
                retry_count += 1
            except:
                time.sleep(0.5)
        time.sleep(0.2)
        return GradeResult.from_oai_response(response)

    def _call_openai_api(self, model: str, messages: Sequence[Mapping[str, str]], max_retries=5):
        """
        Call the OpenAI API with exponential backoff.
        
        Args:
        - model: The model to use for the API call.
        - messages: The messages to send to the API.
        - max_retries: Maximum number of retries.

        Returns:
        - Response from the API or None if all retries failed.
        """
        retry_delays = self._exponential_backoff()
        for _ in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=5
                )
                return response
            except Exception as e:
                # Log the exception if needed
                print(f"An error occurred: {e}")
                time.sleep(next(retry_delays))
        return None


    @staticmethod
    def _exponential_backoff(base_delay=0.5, factor=2, max_delay=60) -> float:
        delay = base_delay
        while True:
            yield delay
            delay = min(delay * factor, max_delay)
