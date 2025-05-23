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
        item: Any,  # Changed from Dict[str, Any] to Any
        question: str,
        llm_client: LLMClient,
        template_formatter: Callable[[str, str], str],
        source_id: str = "unknown"  # Added parameter for source ID
    ) -> Dict[str, Any]:
        """
        Default map function for community summaries.
        
        Args:
            item: The community summary (can be dict or string)
            question: The user question
            llm_client: The LLM client
            template_formatter: Function to format the prompt template
            source_id: Optional source ID to use if item is not a dict
            
        Returns:
            The mapped result with answer and helpfulness
        """
        try:
            # Format the prompt - handles both string and dict inputs
            if isinstance(item, dict):
                prompt = template_formatter(question, item)
                # Extract source ID from item if it's a dict
                source = item.get("id", source_id)
            else:
                # If item is a string, just use it directly
                prompt = template_formatter(question, item)
                source = source_id
            
            # Generate the answer
            response = await llm_client.generate(prompt)
            
            # Parse the helpfulness score
            answer, helpfulness = MapReduceProcessor._parse_helpfulness(response)
            
            # Check if the answer is helpful
            if helpfulness > 0:
                return {
                    "answer": answer,
                    "helpfulness": helpfulness,
                    "source": source
                }
            else:
                return {
                    "answer": "This information is not relevant to the question.",
                    "helpfulness": 0.0,
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
        
        Args:
            mapped_results: List of mapped results
            question: The user question
            llm_client: The LLM client
            template_formatter: Function to format the prompt template
            
        Returns:
            The final answer
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
        
        # Generate the final answer
        prompt = template_formatter(question, formatted_answers)
        
        response = await llm_client.generate(prompt)
        
        return response
    
    @staticmethod
    def _parse_helpfulness(answer: str) -> Tuple[str, int]:
        """
        Parse the helpfulness score from an answer.
        
        Args:
            answer: The generated answer
            
        Returns:
            A tuple of (cleaned_answer, helpfulness_score)
        """
        helpfulness = 0
        cleaned_answer = answer
        
        # Try to extract the helpfulness score
        helpfulness_marker = "<ANSWER HELPFULNESS>"
        helpfulness_end_marker = "</ANSWER HELPFULNESS>"
        
        if helpfulness_marker in answer and helpfulness_end_marker in answer:
            # Extract the score
            start_idx = answer.find(helpfulness_marker) + len(helpfulness_marker)
            end_idx = answer.find(helpfulness_end_marker)
            
            if start_idx < end_idx:
                helpfulness_str = answer[start_idx:end_idx].strip()
                
                try:
                    helpfulness = int(helpfulness_str)
                    
                    # Remove the helpfulness markers from the answer
                    cleaned_answer = (
                        answer[:answer.find(helpfulness_marker)] + 
                        answer[answer.find(helpfulness_end_marker) + len(helpfulness_end_marker):]
                    ).strip()
                except ValueError:
                    # Invalid helpfulness score
                    pass
        
        # Check for JSON format answers that might have a 'helpfulness' field
        import json
        try:
            if '{' in answer and '}' in answer:
                json_start = answer.find('{')
                json_end = answer.rfind('}') + 1
                json_str = answer[json_start:json_end]
                
                data = json.loads(json_str)
                if 'helpfulness' in data:
                    try:
                        helpfulness = float(data['helpfulness'])
                        if 'answer' in data:
                            cleaned_answer = data['answer']
                    except:
                        pass
        except:
            pass
            
        return cleaned_answer, helpfulness