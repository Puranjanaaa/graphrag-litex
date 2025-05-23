"""
Community summarization for GraphRAG.
"""
from typing import Dict, Any
from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from utils.llm_client import LLMClient
from utils.prompts import PromptTemplates
import json
import asyncio

class CommunitySummarizer:
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        self.llm_client = llm_client
        self.config = config
        self.context_window_size = config.context_window_size

    async def summarize_community(
        self,
        community_data: Dict[str, Any],
        kg: KnowledgeGraph,
        community_summaries: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        community_id = community_data["community_id"]
        input_text = self._prepare_input_text(community_data, kg, community_summaries)
        prompt = PromptTemplates.format_community_summary_prompt(input_text)

        try:
            summary_json = await self.llm_client.extract_json(prompt)
            summary = self._clean_summary(summary_json)
            return summary
        except Exception as e:
            return {
                "title": f"Community {community_id}",
                "summary": f"A community with {len(community_data.get('entities', {}))} entities",
                "rating": 5.0,
                "rating explanation": "Neutral importance rating due to summarization failure",
                "findings": [
                    {
                        "summary": "Error in summarization",
                        "explanation": f"Failed to generate a proper summary: {str(e)}"
                    }
                ]
            }

    async def summarize_communities(
        self,
        community_hierarchy: Dict[str, Dict[str, Any]],
        kg: KnowledgeGraph,
        community_detector
    ) -> Dict[str, Dict[str, Any]]:
        community_summaries = {}
        print(f"[DEBUG] Starting community summarization for {len(community_hierarchy)} levels")
        levels = sorted(community_hierarchy.keys(), key=lambda x: int(x[1:]), reverse=True)

        for level_key in levels:
            level_data = community_hierarchy[level_key]
            level_communities = level_data["communities"]
            print(f"[DEBUG] Processing level {level_key} with {len(level_communities)} communities")

            tasks = []
            community_data_list = []

            for community_id, community_info in level_communities.items():
                community_data = community_detector.get_community_entities(
                    kg, community_id, community_hierarchy
                )
                community_data_list.append(community_data)

                task = self.summarize_community(
                    community_data, kg, community_summaries
                )
                tasks.append(task)

            if tasks:
                print(f"[DEBUG] Starting batch processing of {len(tasks)} summarization tasks")

                try:
                    summaries = await asyncio.gather(*tasks)

                    for i, community_id in enumerate([data["community_id"] for data in community_data_list]):
                        summary = summaries[i] if i < len(summaries) else None

                        if not isinstance(summary, dict):
                            print(f"[DEBUG] ERROR - Summary for {community_id} is not a dictionary: {type(summary)}")
                            summary = {
                                "title": f"Community {community_id}",
                                "summary": f"Error processing community data",
                                "rating": 5.0,
                                "rating explanation": "Error in processing",
                                "findings": [{
                                    "summary": "Processing error",
                                    "explanation": f"Expected dictionary, got {type(summary)}"
                                }]
                            }

                        community_summaries[community_id] = summary
                        print(f"[DEBUG] Added summary for community {community_id}: {summary.get('title', 'No title')}")

                        if isinstance(summary, dict):
                            kg.add_community_summary(community_id, summary)
                            print(f"[DEBUG] Added community {community_id} summary to knowledge graph")
                        else:
                            print(f"[DEBUG] Skipped adding summary for {community_id} due to invalid type")

                except Exception as e:
                    import traceback
                    print(f"[DEBUG] CRITICAL ERROR in batch processing: {str(e)}")
                    print(f"[DEBUG] Error traceback: {traceback.format_exc()}")

        print(f"Community summarization complete. Generated {len(community_summaries)} summaries")
        return community_summaries

    def _prepare_input_text(
        self,
        community_data: Dict[str, Any],
        kg: KnowledgeGraph,
        community_summaries: Dict[str, Dict[str, Any]] = None
    ) -> str:
        community_id = community_data["community_id"]
        entities = community_data.get("entities", {})
        relationships = community_data.get("relationships", {})
        claims = community_data.get("claims", {})

        entities_table = "Entities\nid,entity,description\n"
        for entity_id, entity_data in entities.items():
            entities_table += f"{entity_id},{entity_data['name']},{entity_data['description']}\n"

        relationships_table = "Relationships\nid,source,target,description\n"
        for rel_id, rel_data in relationships.items():
            source_id = rel_data["source_id"]
            target_id = rel_data["target_id"]
            source_name = kg.entities[source_id].name if source_id in kg.entities else "Unknown"
            target_name = kg.entities[target_id].name if target_id in kg.entities else "Unknown"
            relationships_table += f"{rel_id},{source_name},{target_name},{rel_data['description']}\n"

        claims_table = "Claims\nid,claim,entities\n"
        for claim_id, claim_data in claims.items():
            entity_names = []
            for entity_id in claim_data["entity_ids"]:
                if entity_id in kg.entities:
                    entity_names.append(kg.entities[entity_id].name)
            entities_str = ",".join(entity_names)
            claims_table += f"{claim_id},{claim_data['content']},{entities_str}\n"

        input_text = f"{entities_table}\n\n{relationships_table}\n\n{claims_table}"

        if community_summaries:
            sub_community_ids = []
            for level_key, level_data in kg.community_summaries.items():
                for summary_id, summary in level_data.items():
                    if isinstance(summary, dict) and summary.get("parent") == community_id:
                        sub_community_ids.append(summary_id)

            if sub_community_ids:
                input_text += "\n\nSub-community Summaries:\n"
                for sub_id in sub_community_ids:
                    if sub_id in community_summaries:
                        sub_summary = community_summaries[sub_id]
                        input_text += f"\n{sub_id}: {sub_summary.get('title', '')}\n"
                        input_text += f"Summary: {sub_summary.get('summary', '')}\n"

        return input_text

    def _clean_summary(self, summary_json: Dict[str, Any]) -> Dict[str, Any]:
        clean_summary = {
            "title": summary_json.get("title", "Untitled Community"),
            "summary": summary_json.get("summary", "No summary provided"),
            "rating": float(summary_json.get("rating", 5.0)),
            "rating explanation": summary_json.get("rating explanation", "No explanation provided"),
            "findings": []
        }

        if not (0 <= clean_summary["rating"] <= 10):
            clean_summary["rating"] = 5.0

        findings = summary_json.get("findings", [])
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict):
                    clean_summary["findings"].append({
                        "summary": finding.get("summary", "No finding summary"),
                        "explanation": finding.get("explanation", "No finding explanation")
                    })

        if not clean_summary["findings"]:
            clean_summary["findings"].append({
                "summary": "Limited information available",
                "explanation": "Insufficient data to generate detailed findings for this community."
            })

        return clean_summary
