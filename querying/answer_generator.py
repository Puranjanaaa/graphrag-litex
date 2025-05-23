"""
Answer generator for GraphRAG.
"""
from typing import List, Dict, Any, Optional
import random
import traceback
from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from utils.llm_client import LLMClient
from utils.prompts import PromptTemplates
from querying.map_reduce import MapReduceProcessor

class AnswerGenerator:
    """
    Generates answers to user queries using community summaries.
    """
    
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        """
        Initialize the answer generator.
        
        Args:
            llm_client: The LLM client
            config: The GraphRAG configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.map_reduce = MapReduceProcessor(llm_client, config)
    
    async def generate_answer(
        self, 
        question: str, 
        kg: KnowledgeGraph,
        community_level: str = "C0"  # Default to root level
    ) -> str:
        """
        Generate an answer to a user question.
        
        Args:
            question: The user question
            kg: The knowledge graph
            community_level: The community level to use
            
        Returns:
            The generated answer
        """
        print(f"Generating answer for question: {question}")
        print(f"Using community level: {community_level}")
        
        # Debug log the structure of community summaries
        if hasattr(kg, 'community_summaries'):
            # print(f"[DEBUG] kg.community_summaries type: {type(kg.community_summaries)}")
            if isinstance(kg.community_summaries, dict):
                # print(f"[DEBUG] kg.community_summaries keys: {list(kg.community_summaries.keys())}")
                total_summaries = sum(
                    1 for _ in kg.community_summaries.keys()
                )
                # print(f"[DEBUG] Total summaries: {total_summaries}")
        
        # Get community summaries for the specified level
        community_summaries = self._get_community_summaries(kg, community_level)
        
        if not community_summaries:
            # print(f"[DEBUG] No summaries found for level {community_level}")
            return "No community summaries available to answer the question."
        
        print(f"Found {len(community_summaries)} summaries to process")
        
        try:
            # Process using map-reduce
            answer = await self.map_reduce.process(
                items=community_summaries,
                question=question,
                map_func=self._map_community_summary,
                reduce_func=self._reduce_community_answers
            )
            
            return answer
        except Exception as e:
            # print(f"[DEBUG] Error in generate_answer: {str(e)}")
            # print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            
            # If map-reduce fails, try a direct approach with the first few summaries
            if community_summaries:
                try:
                    # Use just the first 3 summaries for a simple answer
                    simplified_answer = "Based on the available information:\n\n"
                    
                    for i, summary in enumerate(community_summaries[:3]):
                        title = summary.get('title', f"Community {i+1}")
                        overview = summary.get('summary', "No summary available")
                        simplified_answer += f"• {title}: {overview}\n\n"
                    
                    return simplified_answer
                except:
                    return "Error generating answer with available community summaries."
            else:
                return "No community summaries available to answer the question."
    
    def _get_community_summaries(
        self, 
        kg: KnowledgeGraph,
        community_level: str
    ) -> List[Dict[str, Any]]:
        """
        Get community summaries for a specific level.
        
        Args:
            kg: The knowledge graph
            community_level: The community level
            
        Returns:
            List of community summaries
        """
        level_summaries = []
        
        try:
            # Make sure community_summaries exists
            if not hasattr(kg, 'community_summaries') or kg.community_summaries is None:
                # print(f"[DEBUG] No community_summaries attribute in knowledge graph")
                return []
            
            # Extract level number from community_level (e.g., "C0" -> "0")
            level_num = community_level.replace("C", "")
            
            # Simple direct approach based on the logs
            if isinstance(kg.community_summaries, dict):
                # For level C0 (root), include all summaries from any level
                if level_num == "0":
                    # print(f"[DEBUG] Including all summaries for root level")
                    for community_id, summary in kg.community_summaries.items():
                        # Each key directly maps to a complete summary dictionary
                        if isinstance(summary, dict):
                            # Make a copy and add the ID
                            summary_copy = summary.copy()
                            summary_copy["id"] = community_id
                            level_summaries.append(summary_copy)
                            # print(f"[DEBUG] Added summary for {community_id}: {summary.get('title', 'No title')}")
                else:
                    # For specific levels, only include summaries from that level
                    # e.g., for C2, include '2_0', '2_1', '2_2'
                    level_prefix = f"{level_num}_"
                    for community_id, summary in kg.community_summaries.items():
                        if community_id.startswith(level_prefix):
                            if isinstance(summary, dict):
                                summary_copy = summary.copy()
                                summary_copy["id"] = community_id
                                level_summaries.append(summary_copy)
                                # print(f"[DEBUG] Added summary for {community_id}: {summary.get('title', 'No title')}")
            
            print(f"Found {len(level_summaries)} summaries for level {community_level}")
            
            # Randomize order to reduce positional bias
            if level_summaries:
                random.shuffle(level_summaries)
            
            return level_summaries
            
        except Exception as e:
            # print(f"[DEBUG] Error in _get_community_summaries: {str(e)}")
            # print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            return []
    
    async def _map_community_summary(
        self, 
        community_summary: Dict[str, Any], 
        question: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map a community summary to an answer.
        
        Args:
            community_summary: The community summary
            question: The user question
            
        Returns:
            The mapped result with answer and helpfulness, or None if not helpful
        """
        try:
            community_id = community_summary.get("id", "unknown")
            print(f"Mapping community summary {community_id}")
            
            # Format the summary as report data
            report_data = self._format_summary_as_report(
                community_id, community_summary
            )
            
            # Use the updated default map function with the source_id parameter
            return await MapReduceProcessor.default_map_function(
                item=report_data,
                question=question,
                llm_client=self.llm_client,
                template_formatter=lambda q, r: PromptTemplates.format_community_answer_prompt(
                    question=q, report_data=r
                ),
                source_id=community_id  # Pass the community ID explicitly
            )
        except Exception as e:
            # print(f"[DEBUG] Error in _map_community_summary: {str(e)}")
            # print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            
            # Return a basic result instead of None to avoid breaking the map-reduce flow
            return {
                "answer": f"Error processing community summary {community_summary.get('id', 'unknown')}: {str(e)}",
                "helpfulness": 0.0,
                "source": community_summary.get("id", "unknown")
            }
    
    async def _reduce_community_answers(
        self, 
        mapped_results: List[Dict[str, Any]], 
        question: str
    ) -> str:
        """
        Reduce community answers to a final answer.
        
        Args:
            mapped_results: List of mapped results
            question: The user question
            
        Returns:
            The final answer
        """
        try:
            # print(f"[DEBUG] Reducing {len(mapped_results)} mapped results")
            
            # Filter out None or invalid results
            valid_results = [
                result for result in mapped_results 
                if result is not None and isinstance(result, dict) and "answer" in result
            ]
            
            if not valid_results:
                # print(f"[DEBUG] No valid mapped results to reduce")
                return "No relevant information found to answer the question."
            
            # Sort by helpfulness
            sorted_results = sorted(
                valid_results, 
                key=lambda x: float(x.get("helpfulness", 0)), 
                reverse=True
            )
            
            # print(f"[DEBUG] Processing {len(sorted_results)} valid results")
            
            # Format the community answers
            formatted_answers = ""
            
            for i, result in enumerate(sorted_results):
                formatted_answers += f"\n\nCommunity Answer {i+1}:\n"
                formatted_answers += result.get("answer", "No answer")
            
            # Generate the final answer using the global answer prompt
            prompt = PromptTemplates.format_global_answer_prompt(
                question=question,
                community_answers=formatted_answers
            )
            
            response = await self.llm_client.generate(prompt)
            
            return response
        except Exception as e:
            # print(f"[DEBUG] Error in _reduce_community_answers: {str(e)}")
            # print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            
            # Fallback: return a simple concatenation of the answers
            try:
                simple_answer = "Based on the available information:\n\n"
                
                for i, result in enumerate(mapped_results[:3]):
                    if result and isinstance(result, dict) and "answer" in result:
                        simple_answer += f"• {result.get('answer', 'No answer')}\n\n"
                
                return simple_answer
            except:
                return f"Error generating answer: {str(e)}"
    
    def _format_summary_as_report(self, community_id: str, summary: Dict[str, Any]) -> str:
        """
        Format a community summary as report data.
        
        Args:
            community_id: The community ID
            summary: The community summary
            
        Returns:
            The formatted report data
        """
        try:
            formatted_report = f"Report ID: {community_id}\n"
            formatted_report += f"Title: {summary.get('title', 'Untitled')}\n\n"
            formatted_report += f"Summary: {summary.get('summary', 'No summary')}\n\n"
            formatted_report += f"Impact Rating: {summary.get('rating', 5.0)}\n"
            formatted_report += f"Rating Explanation: {summary.get('rating explanation', 'No explanation')}\n\n"
            
            formatted_report += "Findings:\n"
            findings = summary.get("findings", [])
            if isinstance(findings, list):
                for i, finding in enumerate(findings):
                    if isinstance(finding, dict):
                        formatted_report += f"\nFinding {i+1}: {finding.get('summary', 'No summary')}\n"
                        formatted_report += f"{finding.get('explanation', 'No explanation')}\n"
            
            return formatted_report
        except Exception as e:
            # print(f"[DEBUG] Error formatting summary {community_id}: {str(e)}")
            return f"Error formatting report for {community_id}: {str(e)}"