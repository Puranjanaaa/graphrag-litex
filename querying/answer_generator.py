# ✅ File: querying/answer_generator.py

from typing import List, Dict, Any, Optional
import random
from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from utils.llm_client import LLMClient
from utils.prompts import PromptTemplates
from querying.map_reduce import MapReduceProcessor
from utils.embedding_utils import EmbeddingUtils
import json


class AnswerGenerator:
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
    ) -> dict:
        summaries = self._get_community_summaries(kg, community_level)
        if not summaries:
            return {
                "answer": "No community summaries available to answer the question.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        selected = await self.embedding_util.select_top_k_summaries(
            summaries=summaries,
            question=question,
            k=top_k
        )

        if not selected:
            return {
                "answer": "No relevant community summaries found for the question.",
                "topics": [],
                "used_entities": [],
                "used_relationships": [],
                "used_chunks": []
            }

        result = await self.map_reduce.process(
            items=selected,
            question=question,
            map_func=self._map_community_summary,
            reduce_func=self._reduce_community_answers
        )

        result.setdefault("used_entities", [s.get("id") for s in selected])
        result.setdefault("used_relationships", [])
        result.setdefault("used_chunks", [])

        return result

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
        community_id = community_summary.get("id", "unknown")
        report_data = self._format_summary_as_report(community_id, community_summary)

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

    async def _reduce_community_answers(
        self,
        mapped_results: List[Dict[str, Any]],
        question: str
    ) -> dict:
        sorted_results = sorted(mapped_results, key=lambda x: float(x.get("helpfulness", 0)), reverse=True)
        formatted = ""
        for i, res in enumerate(sorted_results):
            formatted += f"\n\nCommunity Answer {i+1} ({res.get('source', '?')}):\n"
            formatted += res.get("answer", "No answer")

        prompt = (
            f"You are a JSON-generating assistant. Answer the following question using the provided community summaries.\n\n"
            f"Your response MUST be a JSON object with:\n"
            f"- \"answer\" (string): a markdown-formatted summary answer.\n"
            f"- \"topics\" (list): list of topic objects, each with:\n"
            f"    - \"topic\": short title\n"
            f"    - \"description\": brief explanation\n"
            f"    - \"sources\": list of community labels (e.g., [\"Community Answer 1\"])\n"
            f"- \"confidence\" (float, optional)\n\n"
            f"Respond ONLY with valid JSON — no commentary, thinking, or explanations outside the JSON block.\n\n"
            f"Question:\n{question}\n\n"
            f"Community Answers:\n{formatted}"
        )

        response_json = await self.llm_client.extract_json(prompt)

        print("[DEBUG] Full structured reduce response:")
        print(json.dumps(response_json, indent=2))

        answer_text = response_json.get("answer")
        topics = response_json.get("topics")

        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            print("⚠️ [WARNING] LLM returned no valid 'answer'.")
            answer_text = "⚠️ Final structured answer was empty or missing."

        if not isinstance(topics, list):
            print("⚠️ [WARNING] LLM returned invalid or missing 'topics'.")
            topics = []

        return {
            "answer": answer_text,
            "topics": topics,
            "used_entities": [],
            "used_relationships": [],
            "used_chunks": []
        }

    def _format_summary_as_report(self, community_id: str, summary: Dict[str, Any]) -> str:
        findings = summary.get("findings", [])
        finding_text = ""
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
