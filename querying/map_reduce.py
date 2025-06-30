from typing import List, Dict, Any, Tuple, Callable, TypeVar, Awaitable
import logging
import traceback

from config import GraphRAGConfig
from utils.llm_client import LLMClient
from utils.async_utils import map_async

T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger("MapReduceProcessor")


class MapReduceProcessor:
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        self.llm_client = llm_client
        self.config = config
        self.context_window_size = config.context_window_size

    async def process(
        self,
        items: List[T],
        question: str,
        map_func: Callable[[T, str], Awaitable[R]],
        reduce_func: Callable[[List[R], str], Awaitable[Dict[str, Any]]],
        max_concurrency: int = 5
    ) -> Dict[str, Any]:
        mapped_results = await map_async(
            items,
            lambda item: map_func(item, question),
            max_concurrency=max_concurrency
        )

        filtered_results = [result for result in mapped_results if result and result.get("answer")]

        if not filtered_results:
            logger.warning("No relevant results after mapping. Returning default empty answer.")
            return {
                "answer": "No relevant information found to answer the question.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        final_answer = await reduce_func(filtered_results, question)

        used_entities = [result.get("source") for result in filtered_results if result.get("source")]
        used_relationships = []
        used_chunks = []

        for result in filtered_results:
            used_relationships.extend(result.get("used_relationships", []))
            used_chunks.extend(result.get("used_chunks", []))

        final_answer["used_entities"] = used_entities
        final_answer["used_relationships"] = used_relationships
        final_answer["used_chunks"] = used_chunks

        return final_answer

    @staticmethod
    async def default_map_function(
        item: Any,
        question: str,
        llm_client: LLMClient,
        template_formatter: Callable[[str, str], str],
        source_id: str = "unknown"
    ) -> Dict[str, Any]:
        try:
            if isinstance(item, dict):
                prompt = template_formatter(question, item)
                source = item.get("id", source_id)
            else:
                prompt = template_formatter(question, item)
                source = source_id

            response_json = await llm_client.extract_json(prompt)

            answer = response_json.get("answer", "")
            helpfulness = float(response_json.get("helpfulness", 0))

            return {
                "answer": answer,
                "helpfulness": helpfulness,
                "source": source,
                "used_relationships": response_json.get("used_relationships", []),
                "used_chunks": response_json.get("used_chunks", [])
            }

        except Exception as e:
            logger.error(f"Error in default_map_function: {str(e)}")
            return {
                "answer": f"Error processing item: {str(e)}",
                "helpfulness": 0.0,
                "source": source_id,
                "used_relationships": [],
                "used_chunks": []
            }

    @staticmethod
    async def default_reduce_function(
        mapped_results: List[Dict[str, Any]],
        question: str,
        llm_client: LLMClient,
        template_formatter: Callable[[str, str], str]
    ) -> Dict[str, Any]:
        if not mapped_results:
            logger.warning("No mapped results provided to reduce function.")
            return {
                "answer": "No relevant information found to answer the question.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        sorted_results = sorted(mapped_results, key=lambda x: x.get("helpfulness", 0), reverse=True)

        formatted_answers = ""
        for i, result in enumerate(sorted_results):
            formatted_answers += f"\n\nCommunity Answer {i+1} ({result.get('source', '?')}):\n"
            formatted_answers += result.get("answer", "No answer")

        prompt = (
            f"You are a helpful assistant summarizing answers to the following question:\n"
            f"### Question:\n{question}\n\n"
            f"### Task:\n"
            f"Summarize the key findings from the community responses below in a markdown-formatted answer.\n"
            f"\nReturn only a **strictly valid JSON** object with the following keys:\n"
            f"1. \"answer\": (required) A markdown-formatted answer that summarizes the findings related to the question.\n"
            f"2. \"topics\": (required) A list of key topics, each containing:\n"
            f"    - topic: title of the topic.\n"
            f"    - description: a short explanation (not 'details').\n"
            f"    - sources: a list of community labels (e.g., [\"Community Answer 1\"]).\n"
            f"3. \"confidence\": (optional) A float between 0 and 1 estimating confidence.\n"
            f"\nStrictly use these field names: \"answer\", \"topics\", \"sources\".\n"
            f"DO NOT include keys like \"main_topics\", \"details\", or any text outside the JSON block.\n"
            f"\n### Community Responses:\n{formatted_answers}"
        )

        try:
            response_json = await llm_client.extract_json(prompt)

            if not all(k in response_json for k in ["answer", "topics"]):
                raise ValueError("Missing required keys in structured answer")

            return {
                "answer": response_json.get("answer", ""),
                "topics": response_json.get("topics", []),
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        except Exception as e:
            logger.error(f"Error parsing structured reduce output: {e}")
            return {
                "answer": "⚠️ Failed to parse a structured summary from the LLM. Please inspect community-level answers.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }
