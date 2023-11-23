import os
from typing import Any, Mapping, Optional, Sequence

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

from .core import Model

load_dotenv()


class HuggingfaceCompletionRequest(BaseModel):
    """Contains a couple of useful parameters offered in the Huggingface API."""

    inputs: str
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: float = 1.0
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    max_time: Optional[float] = None
    return_full_text: bool = False  # Hf default: True

    def to_json(self) -> Mapping[str, Any]:
        return {
            "inputs": self.inputs,
            "parameters": {
                "top_k": self.top_k,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens,
                "max_time": self.max_time,
                "return_full_text": self.return_full_text,
            },
        }


class HuggingfaceClient:
    def __init__(self, model: Model) -> None:
        token = os.getenv("HF_ACCESS_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.url = f"https://api-inference.huggingface.co/models/{model.value}"
        self.tokenizer = AutoTokenizer.from_pretrained(model.value, token=token)

    def complete(self, request: HuggingfaceCompletionRequest) -> str:
        response = requests.post(self.url, headers=self.headers, json=request.to_json())
        response.raise_for_status()
        json_response: Sequence[Mapping[str, Any]] = response.json()
        completion = json_response[0].get("generated_text")
        if isinstance(completion, str):
            return completion
        raise TypeError(f"Expected object to be of type str, got {type(completion)}.")

    def messages_to_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Convert a messages object (OpenAI-format) into a prompt string.
        
        For some models, like Mistral's, the first message must be by the user.
        Others, like Llama2, allow for system prompts.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False)  # type: ignore
