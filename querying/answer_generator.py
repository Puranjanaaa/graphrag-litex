"""
Answer generator for GraphRAG.
"""
from typing import List, Dict, Any, Optional
import random
from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from utils.llm_client import LLMClient
from utils.prompts import PromptTemplates
from querying.map_reduce import MapReduceProcessor
from utils.embedding_utils import EmbeddingUtils


class AnswerGenerator:
    """
    Generates answers to user queries using community summaries.
    """

    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        self.llm_client = llm_client
        self.config = config
        self.map_reduce = MapReduceProcessor(llm_client, config)
        self.embedding_util = EmbeddingUtils(config)

    async def generate_answer(
        self,
        question: str,
        kg: KnowledgeGraph,
        community_level: str = "C0",
        top_k: int = 10
    ) -> str:
        print(f"Generating answer for question: {question}")
        print(f"Using community level: {community_level}")

        summaries = self._get_community_summaries(kg, community_level)
        if not summaries:
            return "No community summaries available to answer the question."

        print(f"Retrieved {len(summaries)} total summaries")

        selected = await self.embedding_util.select_top_k_summaries(
            summaries=summaries,
            question=question,
            k=top_k
        )

        print(f"Selected top {len(selected)} summaries for reasoning")

        if not selected:
            return "No relevant community summaries found for the question."

        try:
            return await self.map_reduce.process(
                items=selected,
                question=question,
                map_func=self._map_community_summary,
                reduce_func=self._reduce_community_answers
            )
        except Exception as e:
            print(f"[ERROR] MapReduce failed: {str(e)}")
            return self._fallback_simple_answer(selected)

    def _get_community_summaries(
        self,
        kg: KnowledgeGraph,
        community_level: str
    ) -> List[Dict[str, Any]]:
        summaries = []
        if not hasattr(kg, 'community_summaries') or not isinstance(kg.community_summaries, dict):
            return []

        level = community_level.replace("C", "")
        for community_id, summary in kg.community_summaries.items():
            if level == "0" or community_id.startswith(f"{level}_"):
                if isinstance(summary, dict):
                    summary_copy = summary.copy()
                    summary_copy["id"] = community_id
                    summaries.append(summary_copy)

        random.shuffle(summaries)
        return summaries

    async def _map_community_summary(
        self,
        community_summary: Dict[str, Any],
        question: str
    ) -> Optional[Dict[str, Any]]:
        try:
            community_id = community_summary.get("id", "unknown")
            report_data = self._format_summary_as_report(community_id, community_summary)

            # Use extract_json instead of generate + parsing
            prompt = PromptTemplates.format_community_answer_prompt(
                question=question,
                report_data=report_data
            )
            json_response = await self.llm_client.extract_json(prompt)

            return {
                "answer": json_response.get("answer", ""),
                "helpfulness": json_response.get("helpfulness", 0),
                "source": community_id
            }
        except Exception as e:
            return {
                "answer": f"Error processing summary {community_summary.get('id', 'unknown')}: {str(e)}",
                "helpfulness": 0.0,
                "source": community_summary.get("id", "unknown")
            }

    async def _reduce_community_answers(
        self,
        mapped_results: List[Dict[str, Any]],
        question: str
    ) -> str:
        valid = [r for r in mapped_results if r and "answer" in r]
        if not valid:
            return "No relevant information found to answer the question."

        sorted_results = sorted(valid, key=lambda x: float(x.get("helpfulness", 0)), reverse=True)

        formatted = ""
        for i, res in enumerate(sorted_results):
            formatted += f"\n\nCommunity Answer {i+1} ({res.get('source', '?')}):\n"
            formatted += res.get("answer", "No answer")

        prompt = PromptTemplates.format_global_answer_prompt(
            question=question,
            community_answers=formatted
        )
        return await self.llm_client.generate(prompt)

    def _format_summary_as_report(self, community_id: str, summary: Dict[str, Any]) -> str:
        try:
            findings = summary.get("findings", [])
            finding_text = ""
            if isinstance(findings, list):
                for i, f in enumerate(findings):
                    if isinstance(f, dict):
                        finding_text += f"\nFinding {i+1}: {f.get('summary', '')}\n{f.get('explanation', '')}\n"

            return (
                f"Report ID: {community_id}\n"
                f"Title: {summary.get('title', 'Untitled')}\n\n"
                f"Summary: {summary.get('summary', 'No summary')}\n\n"
                f"Impact Rating: {summary.get('rating', 5.0)}\n"
                f"Rating Explanation: {summary.get('rating explanation', 'No explanation')}\n\n"
                f"Findings:\n{finding_text}"
            )
        except Exception as e:
            return f"Error formatting report {community_id}: {str(e)}"

    def _fallback_simple_answer(self, summaries: List[Dict[str, Any]]) -> str:
        simple_answer = "Based on the available information:\n\n"
        for i, summary in enumerate(summaries[:3]):
            simple_answer += f"• {summary.get('title', f'Community {i+1}')}: {summary.get('summary', 'No summary')}\n\n"
        return simple_answer


async def generate_answer(
    question: str,
    kg: KnowledgeGraph,
    community_level: str = "C0",
    top_k: int = 10
) -> str:
    """
    Standalone function for generating answers.
    This is a wrapper around the AnswerGenerator class for compatibility.
    """
    from config import GraphRAGConfig
    
    config = GraphRAGConfig()
    llm_client = LLMClient()
    generator = AnswerGenerator(llm_client, config)
    
    return await generator.generate_answer(question, kg, community_level, top_k)
