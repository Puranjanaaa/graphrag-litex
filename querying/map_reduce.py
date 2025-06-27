### File: querying/map_reduce.py

from typing import List, Dict, Any, Tuple, Callable, TypeVar, Awaitable
from config import GraphRAGConfig
from utils.llm_client import LLMClient
from utils.async_utils import map_async

T = TypeVar('T')
R = TypeVar('R')

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
            return {
                "answer": "No relevant information found to answer the question.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        final_answer = await reduce_func(filtered_results, question)

        used_entities = [result.get("source") for result in filtered_results if result.get("source")]
        final_answer["used_entities"] = used_entities
        final_answer.setdefault("used_relationships", [])
        final_answer.setdefault("used_chunks", [])

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

            answer = response_json.get("answer", "No answer returned")
            helpfulness = float(response_json.get("helpfulness", 0))

            return {
                "answer": answer,
                "helpfulness": helpfulness,
                "source": source
            }

        except Exception as e:
            import traceback
            print(f"[DEBUG] Error in default_map_function: {str(e)}")
            print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            return {
                "answer": f"Error processing item: {str(e)}",
                "helpfulness": 0.0,
                "source": source_id
            }

    @staticmethod
    async def default_reduce_function(
        mapped_results: List[Dict[str, Any]],
        question: str,
        llm_client: LLMClient,
        template_formatter: Callable[[str, str], str]
    ) -> Dict[str, Any]:
        if not mapped_results:
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
            formatted_answers += f"\n\nAnswer {i+1} (Helpfulness: {result.get('helpfulness', 0)}):\n"
            formatted_answers += result.get("answer", "No answer")

        prompt = (
            f"You're an assistant. Return the following JSON structure only:\n"
            f"{{\n"
            f"  \"question\": \"{question}\",\n"
            f"  \"main_topics\": [\"<topic1>\", \"<topic2>\", ...],\n"
            f"  \"summary\": \"<markdown summary answer>\",\n"
            f"  \"confidence\": 0.0 - 1.0\n"
            f"}}\n\n"
            f"Based only on these community answers:\n{formatted_answers}\n"
        )

        try:
            response_json = await llm_client.extract_json(prompt)

            if not all(k in response_json for k in ["question", "main_topics", "summary", "confidence"]):
                raise ValueError("Missing required keys in structured answer")

            return {
                "answer": response_json.get("summary", ""),
                "topics": response_json.get("main_topics", []),
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        except Exception as e:
            import traceback
            print(f"[DEBUG] Error parsing structured reduce output: {e}")
            print(traceback.format_exc())
            return {
                "answer": "⚠️ Failed to parse a structured summary from the LLM. Please inspect community-level answers.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }
