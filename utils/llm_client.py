"""
LLM API wrapper for interacting with language models.
"""
import os
import json
import asyncio
import backoff
from typing import Dict, Any, List, Optional, Union

import aiohttp
import logging

from config import GraphRAGConfig

from dotenv import load_dotenv


load_dotenv()  # Load variables from .env file
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")

class LLMClient:
    """
    A client for interacting with locally hosted language models through LM Studio.
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the LLM client for a locally hosted model via LM Studio.
        
        Args:
            config: GraphRAG configuration
        """
        print(f"Initializing LLMClient")
        self.config = config
        self.base_url = os.getenv("LM_STUDIO_URL")
        print(f"Using base URL: {self.base_url}")
        self.logging_enabled = config.logging_enabled
        
        if self.logging_enabled:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("LLMClient")
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        base=2
    )
    async def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from the locally hosted model via LM Studio.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use (overrides config, not used for local models)
            temperature: The temperature to use (overrides config)
            max_tokens: The maximum number of tokens to generate (overrides config)
            
        Returns:
            The generated text
        """
        # print(f"[DEBUG] generate() called with prompt length: {len(prompt)}")
        # print(f"[DEBUG] model: {model}, temperature: {temperature}, max_tokens: {max_tokens}")
        
        # Use fallback values from config if not specified
        temperature = temperature if temperature is not None else self.config.llm_temperature
        max_tokens = max_tokens or self.config.llm_max_tokens
        # print(f"[DEBUG] Using temperature: {temperature}, max_tokens: {max_tokens}")
        
        # Log the request if logging is enabled
        if self.logging_enabled:
            self.logger.info(f"Sending prompt to local model (first 100 chars): {prompt[:100]}...")
        
        # Prepare the request payload for LM Studio's API
        # LM Studio follows the OpenAI API format
        payload = {
            "model": model or "local-model",  # Model name is often ignored for local deployments
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        # print(f"[DEBUG] Request payload prepared: {json.dumps(payload)[:200]}...")
        
        # Make the API request
        # print(f"[DEBUG] Making API request to: {self.base_url}/chat/completions")
        try:
            async with aiohttp.ClientSession() as session:
                # print(f"[DEBUG] Session created")
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.config.llm_timeout
                ) as response:
                    # print(f"[DEBUG] Response status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        # print(f"[DEBUG] Error response: {error_text}")
                        raise ValueError(f"Error from LM Studio API: {response.status}, {error_text}")
                    
                    response_json = await response.json()
                    # print(f"[DEBUG] Response JSON received: {str(response_json)[:200]}...")
                    
                    # Extract the response text
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        result = response_json["choices"][0]["message"]["content"]
                        # print(f"[DEBUG] Successfully extracted result (length: {len(result)})")
                        
                        # Log the response if logging is enabled
                        if self.logging_enabled:
                            self.logger.info(f"Received response from local model (first 100 chars): {result[:100]}...")
                        
                        return result
                    else:
                        # print(f"[DEBUG] Invalid response format: {response_json}")
                        raise ValueError(f"Invalid response format from LM Studio API: {response_json}")
        except Exception as e:
            # print(f"[DEBUG] Exception during API request: {str(e)}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        base=2
    )
    async def batch_generate(
        self, 
        prompts: List[str], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate multiple texts from the language model in parallel.
        
        Args:
            prompts: The prompts to send to the model
            model: The model to use (overrides config)
            temperature: The temperature to use (overrides config)
            max_tokens: The maximum number of tokens to generate (overrides config)
            
        Returns:
            A list of generated texts
        """
        # print(f"[DEBUG] batch_generate() called with {len(prompts)} prompts")
        
        tasks = []
        for i, prompt in enumerate(prompts):
            # print(f"[DEBUG] Creating task for prompt {i+1}/{len(prompts)}, length: {len(prompt)}")
            task = self.generate(prompt, model, temperature, max_tokens)
            tasks.append(task)
        
        # print(f"[DEBUG] Gathering {len(tasks)} tasks")
        results = await asyncio.gather(*tasks)
        # print(f"[DEBUG] All {len(results)} tasks completed")
        return results
    
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
        """
        Extract structured JSON data from the language model.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use (overrides config)
            temperature: The temperature to use (overrides config)
            max_tokens: The maximum number of tokens to generate (overrides config)
            
        Returns:
            The extracted JSON data
        """
        # print(f"[DEBUG] extract_json() called with prompt length: {len(prompt)}")
        
        # For JSON extraction, we'll use a lower temperature for more deterministic output
        json_temperature = temperature if temperature is not None else max(0.1, self.config.llm_temperature / 2)
        # print(f"[DEBUG] Using json_temperature: {json_temperature}")
        
        # Add explicit instructions for JSON output
        enhanced_prompt = (
            prompt + 
            "\n\nYour response must be a valid, parseable JSON object. " +
            "Do not include any explanations or text outside of the JSON object. " +
            "Format your entire response as a single JSON object."
        )
        # print(f"[DEBUG] Enhanced prompt length: {len(enhanced_prompt)}")
        
        # First, try to get a response from the model
        # print("[DEBUG] Calling generate() for JSON extraction")
        response_text = await self.generate(
            prompt=enhanced_prompt,
            model=model,
            temperature=json_temperature,
            max_tokens=max_tokens
        )
        # print(f"[DEBUG] Received response for JSON extraction, length: {len(response_text)}")
        # print(f"[DEBUG] Response text preview: {response_text[:100]}...")
        
        # Try to parse the response as JSON
        try:
            # print("[DEBUG] Attempting to parse JSON response")
            # Find JSON-like content in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            # print(f"[DEBUG] JSON markers found at positions: start={json_start}, end={json_end}")
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end+1]
                # print(f"[DEBUG] Extracted JSON content, length: {len(json_content)}")
                parsed_json = json.loads(json_content)
                # print(f"[DEBUG] Successfully parsed JSON with {len(parsed_json)} keys")
                return parsed_json
            
            # Try to load the response directly if it doesn't have obvious JSON markers
            # print("[DEBUG] No clear JSON markers, trying to parse entire response")
            parsed_json = json.loads(response_text)
            # print(f"[DEBUG] Successfully parsed direct JSON with {len(parsed_json)} keys")
            return parsed_json
        except json.JSONDecodeError as e:
            # print(f"[DEBUG] JSON parsing error: {str(e)}")
            # If parsing fails, try to get a more structured response
            if self.logging_enabled:
                self.logger.warning(f"JSON parsing error: {str(e)}. Retrying with more explicit instructions.")
            
            retry_prompt = (
                "The previous response couldn't be parsed as JSON. "
                "Please format your response as a valid JSON object with no additional text.\n\n"
                "Original request:\n" + prompt
            )
            
            # print("[DEBUG] Retrying with more explicit instructions")
            response_text = await self.generate(
                prompt=retry_prompt,
                model=model,
                temperature=0.0,  # Use zero temperature for most deterministic output
                max_tokens=max_tokens
            )
            # print(f"[DEBUG] Received retry response, length: {len(response_text)}")
            
            # Try again to parse the response
            try:
                # print("[DEBUG] Attempting to parse retry JSON response")
                # Find JSON-like content in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                # print(f"[DEBUG] Retry JSON markers found at positions: start={json_start}, end={json_end}")
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end+1]
                    # print(f"[DEBUG] Extracted retry JSON content, length: {len(json_content)}")
                    parsed_json = json.loads(json_content)
                    # print(f"[DEBUG] Successfully parsed retry JSON with {len(parsed_json)} keys")
                    return parsed_json
                
                # Try to load the response directly if it doesn't have obvious JSON markers
                # print("[DEBUG] No clear retry JSON markers, trying to parse entire response")
                parsed_json = json.loads(response_text)
                # print(f"[DEBUG] Successfully parsed direct retry JSON with {len(parsed_json)} keys")
                return parsed_json
            except json.JSONDecodeError:
                # print("[DEBUG] Failed to parse JSON response after retry")
                # If all else fails, return a simple error message as JSON
                if self.logging_enabled:
                    self.logger.error(f"Failed to parse JSON response after retry. Raw response: {response_text[:500]}...")
                
                return {
                    "error": "Failed to parse response as JSON", 
                    "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
                }