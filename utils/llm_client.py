"""
LLM API wrapper for interacting with language models.
"""
import os
import json
import asyncio
import backoff
from typing import Dict, Any, List, Optional

import aiohttp
import logging

from config import GraphRAGConfig
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")


class LLMClient:
    """
    A client for interacting with locally hosted language models through LM Studio.
    """

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.base_url = os.getenv("LM_STUDIO_URL")
        self.logging_enabled = config.logging_enabled
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if self.logging_enabled:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("LLMClient")

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        base=2
    )
    async def embed(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embedding_model.encode, text)

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        temperature = temperature if temperature is not None else self.config.llm_temperature
        max_tokens = max_tokens or self.config.llm_max_tokens

        if self.logging_enabled:
            self.logger.info(f"Sending prompt to local model (first 100 chars): {prompt[:100]}...")

        payload = {
            "model": model or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.config.llm_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error from LM Studio API: {response.status}, {error_text}")

                    response_json = await response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        result = response_json["choices"][0]["message"]["content"]

                        if self.logging_enabled:
                            self.logger.info(f"Received response (first 100 chars): {result[:100]}...")

                        return result
                    raise ValueError(f"Invalid response format: {response_json}")
        except Exception as e:
            raise

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError),
        max_tries=5,
        base=2
    )
    async def extract_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        json_temperature = temperature if temperature is not None else max(0.1, self.config.llm_temperature / 2)

        # Enforce strict JSON-only response
        enhanced_prompt = (
            prompt.strip() +
            "\n\nYour response must be a valid, parseable JSON object. " +
            "Do not include any explanations or text outside of the JSON object."
        )

        response_text = await self.generate(
            prompt=enhanced_prompt,
            model=model,
            temperature=json_temperature,
            max_tokens=max_tokens
        )

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end + 1])
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            if self.logging_enabled:
                self.logger.warning(f"Initial JSON parse failed: {e}. Retrying with simplified prompt.")

            retry_prompt = (
                "Please return the following as a valid JSON object with no additional text.\n\n" +
                prompt.strip()
            )
            retry_response = await self.generate(
                prompt=retry_prompt,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens
            )
            try:
                retry_start = retry_response.find('{')
                retry_end = retry_response.rfind('}')
                if retry_start >= 0 and retry_end > retry_start:
                    return json.loads(retry_response[retry_start:retry_end + 1])
                return json.loads(retry_response)
            except json.JSONDecodeError:
                if self.logging_enabled:
                    self.logger.error(f"Retry failed to parse JSON. Raw output: {retry_response[:300]}")
                return {
                    "error": "Failed to parse response as JSON",
                    "raw_response": retry_response[:500] + ("..." if len(retry_response) > 500 else "")
                }

    async def generate_structured_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wrapper around extract_json, used for generating structured final answers.
        """
        return await self.extract_json(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
