"""
Map-reduce for community answers in GraphRAG.
"""
from typing import List, Dict, Any, Tuple, Callable, TypeVar, Awaitable
from config import GraphRAGConfig
from utils.llm_client import LLMClient
from utils.async_utils import map_async

T = TypeVar('T')
R = TypeVar('R')

class MapReduceProcessor:
    """
    Processes queries using a map-reduce approach.
    """
    
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        """
        Initialize the map-reduce processor.
        
        Args:
            llm_client: The LLM client
            config: The GraphRAG configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.context_window_size = config.context_window_size
    
    async def process(
        self,
        items: List[T],
        question: str,
        map_func: Callable[[T, str], Awaitable[R]],
        reduce_func: Callable[[List[R], str], Awaitable[str]],
        max_concurrency: int = 5
    ) -> str:
        """
        Process items using a map-reduce approach.
        
        Args:
            items: List of items to process
            question: The user question
            map_func: Function to map an item to an answer
            reduce_func: Function to reduce answers to a final answer
            max_concurrency: Maximum number of concurrent tasks
            
        Returns:
            The final answer
        """
        # Map phase: Process each item in parallel
        mapped_results = await map_async(
            items,
            lambda item: map_func(item, question),
            max_concurrency=max_concurrency
        )
        
        # Filter out empty or unhelpful results
        filtered_results = [result for result in mapped_results if result]
        
        if not filtered_results:
            return "No relevant information found to answer the question."
        
        # Reduce phase: Combine all results into a final answer
        final_answer = await reduce_func(filtered_results, question)
        
        return final_answer
    
    @staticmethod
    async def default_map_function(
        item: Any,
        question: str,
        llm_client: LLMClient,
        template_formatter: Callable[[str, str], str],
        source_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Default map function for community summaries.
        """
        try:
            if isinstance(item, dict):
                prompt = template_formatter(question, item)
                source = item.get("id", source_id)
            else:
                prompt = template_formatter(question, item)
                source = source_id

            # Use robust JSON extraction
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
    ) -> str:
        """
        Default reduce function for community answers.
        Enforces structured output (JSON).
        """
        if not mapped_results:
            return "No relevant information found to answer the question."
        
        # Sort by helpfulness
        sorted_results = sorted(mapped_results, key=lambda x: x.get("helpfulness", 0), reverse=True)
        
        # Format the community answers
        formatted_answers = ""
        
        for i, result in enumerate(sorted_results):
            formatted_answers += f"\n\nAnswer {i+1} (Helpfulness: {result.get('helpfulness', 0)}):\n"
            formatted_answers += result.get("answer", "No answer")
        
        # Generate the structured response
        prompt = template_formatter(question, formatted_answers)

        try:
            response_json = await llm_client.extract_json(prompt)

            # Ensure required fields are present
            if not all(k in response_json for k in ["question", "main_topics", "summary", "confidence"]):
                raise ValueError("Missing required keys in structured answer")

            return response_json  # Final structured result as JSON/dict

        except Exception as e:
            import traceback
            print(f"[DEBUG] Error parsing structured reduce output: {e}")
            print(traceback.format_exc())
            return {
                "question": question,
                "main_topics": [],
                "summary": "Failed to generate a structured response.",
                "confidence": 0.0
            }

    @staticmethod
    def _parse_helpfulness(answer: str) -> Tuple[str, int]:
        """
        [Deprecated] Helper to parse helpfulness from non-JSON answers.
        Not used anymore.
        """
        return answer, 0  # Not used since extract_json is now the default
