import os
import json
import time
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import openai


@dataclass
class InferenceRecord:
    query_id: str
    method: str
    model: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    temperature: float = 0.0
    seed: Optional[int] = None
    finish_reason: str = ""

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "method": self.method,
            "model": self.model,
            "raw_response": self.raw_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "temperature": self.temperature,
            "seed": self.seed,
            "finish_reason": self.finish_reason,
        }


class MLLMClient:
    def __init__(self, model: str = "gpt-4-vision-preview", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int = 42, top_p: float = 1.0,
                 api_key_env: str = "OPENAI_API_KEY", max_retries: int = 3,
                 retry_delay: float = 5.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        self.client = openai.OpenAI(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _prepare_messages(self, messages: list) -> list:
        prepared = []
        for msg in messages:
            role = msg["role"]
            content_items = msg["content"]
            prepared_content = []
            for item in content_items:
                if item["type"] == "text":
                    prepared_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    img_path = item["image_url"]["url"]
                    if os.path.exists(img_path):
                        b64 = self._encode_image(img_path)
                        ext = Path(img_path).suffix.lower().replace(".", "")
                        if ext == "jpg":
                            ext = "jpeg"
                        prepared_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                        })
                    else:
                        prepared_content.append(item)
            prepared.append({"role": role, "content": prepared_content})
        return prepared

    def infer(self, messages: list, query_id: str = "", method: str = "") -> InferenceRecord:
        prepared = self._prepare_messages(messages)

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prepared,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    top_p=self.top_p,
                )
                elapsed = (time.time() - start) * 1000

                choice = response.choices[0]
                return InferenceRecord(
                    query_id=query_id,
                    method=method,
                    model=self.model,
                    raw_response=choice.message.content,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=elapsed,
                    temperature=self.temperature,
                    seed=self.seed,
                    finish_reason=choice.finish_reason,
                )
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def infer_batch(self, batch: list, delay: float = 0.5) -> list:
        results = []
        for item in batch:
            result = self.infer(
                messages=item["messages"],
                query_id=item.get("query_id", ""),
                method=item.get("method", ""),
            )
            results.append(result)
            if delay > 0:
                time.sleep(delay)
        return results
