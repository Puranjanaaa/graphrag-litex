"""
Query processor for GraphRAG.
"""
from typing import List, Dict, Any, Tuple
import asyncio
import random
from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from utils.llm_client import LLMClient
from utils.prompts import PromptTemplates

class QueryProcessor:
    """
    Processes user queries against the knowledge graph.
    """
    
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        """
        Initialize the query processor.
        
        Args:
            llm_client: The LLM client
            config: The GraphRAG configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.context_window_size = config.context_window_size
    
    async def process_query(
        self, 
        query: str, 
        kg: KnowledgeGraph,
        community_level: str = "C0"  # Default to root level
    ) -> str:
        """
        Process a user query.
        
        Args:
            query: The user query
            kg: The knowledge graph
            community_level: The community level to use
            
        Returns:
            The answer to the query
        """
        # Get community summaries for the specified level
        level_summaries = self._get_level_summaries(kg, community_level)
        
        if not level_summaries:
            return "No community summaries available to answer the query."
        
        # Get community answers
        community_answers = await self._get_community_answers(query, level_summaries)
        
        # Generate the final answer
        final_answer = await self._generate_global_answer(query, community_answers)
        
        return final_answer
    
    def _get_level_summaries(
        self, 
        kg: KnowledgeGraph,
        community_level: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get community summaries for a specific level.
        
        Args:
            kg: The knowledge graph
            community_level: The community level
            
        Returns:
            Dictionary of community summaries
        """
        level_summaries = {}
        
        # Get all community summaries
        community_summaries = kg.community_summaries
        
        # Filter to those in the specified level
        for community_id, summary in community_summaries.items():
            # Check if the community is at the specified level
            if community_id.startswith(community_level.replace("C", "")):
                level_summaries[community_id] = summary
        
        return level_summaries
    
    async def _get_community_answers(
        self, 
        query: str, 
        summaries: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get answers from individual communities.
        
        Args:
            query: The user query
            summaries: Dictionary of community summaries
            
        Returns:
            List of community answers
        """
        tasks = []
        
        # Randomize the order of summaries to reduce positional bias
        summary_items = list(summaries.items())
        random.shuffle(summary_items)
        
        # Create a batch of tasks
        for community_id, summary in summary_items:
            # Format the summary as report data
            report_data = self._format_summary_as_report(community_id, summary)
            
            # Create a task to generate an answer
            task = self._generate_community_answer(query, report_data)
            tasks.append((community_id, task))
        
        # Process in batches
        all_answers = []
        
        # Process tasks in batches
        batch_size = 5
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_ids = [community_id for community_id, _ in batch]
            batch_tasks = [task for _, task in batch]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            for j, result in enumerate(batch_results):
                community_id = batch_ids[j]
                
                # Parse the helpfulness score from the result
                answer, helpfulness = self._parse_helpfulness(result)
                
                # Check if the answer is helpful
                if helpfulness > 0:
                    all_answers.append({
                        "community_id": community_id,
                        "answer": answer,
                        "helpfulness": helpfulness
                    })
        
        # Sort answers by helpfulness (descending)
        all_answers.sort(key=lambda x: x["helpfulness"], reverse=True)
        
        # Take the top answers up to the maximum
        return all_answers[:self.config.max_community_answers]
    
    async def _generate_community_answer(self, query: str, report_data: str) -> str:
        """
        Generate an answer from a community.
        
        Args:
            query: The user query
            report_data: The formatted report data
            
        Returns:
            The generated answer
        """
        prompt = PromptTemplates.format_community_answer_prompt(
            question=query,
            report_data=report_data
        )
        
        response = await self.llm_client.generate(prompt)
        
        return response
    
    async def _generate_global_answer(self, query: str, community_answers: List[Dict[str, Any]]) -> str:
        """
        Generate a global answer by combining community answers.
        
        Args:
            query: The user query
            community_answers: List of community answers
            
        Returns:
            The global answer
        """
        if not community_answers:
            return "No relevant information found to answer the query."
        
        # Format the community answers
        formatted_answers = ""
        
        for i, answer_data in enumerate(community_answers):
            formatted_answers += f"\n\nCommunity Answer {i+1} (Helpfulness: {answer_data['helpfulness']}):\n"
            formatted_answers += answer_data["answer"]
        
        # Generate the global answer
        prompt = PromptTemplates.format_global_answer_prompt(
            question=query,
            community_answers=formatted_answers
        )
        
        response = await self.llm_client.generate(prompt)
        
        return response
    
    def _format_summary_as_report(self, community_id: str, summary: Dict[str, Any]) -> str:
        """
        Format a community summary as report data.
        
        Args:
            community_id: The community ID
            summary: The community summary
            
        Returns:
            The formatted report data
        """
        formatted_report = f"Report ID: {community_id}\n"
        formatted_report += f"Title: {summary.get('title', 'Untitled')}\n\n"
        formatted_report += f"Summary: {summary.get('summary', 'No summary')}\n\n"
        formatted_report += f"Impact Rating: {summary.get('rating', 5.0)}\n"
        formatted_report += f"Rating Explanation: {summary.get('rating explanation', 'No explanation')}\n\n"
        
        formatted_report += "Findings:\n"
        for i, finding in enumerate(summary.get("findings", [])):
            formatted_report += f"\nFinding {i+1}: {finding.get('summary', 'No summary')}\n"
            formatted_report += f"{finding.get('explanation', 'No explanation')}\n"
        
        return formatted_report
    
    def _parse_helpfulness(self, answer: str) -> Tuple[str, int]:
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
        
        return cleaned_answer, helpfulness