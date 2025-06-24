from typing import List, Dict
import numpy as np
import asyncio
from utils.llm_client import LLMClient
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingUtils:
    """
    Utility class to compute and compare embeddings for semantic search over community summaries.
    """

    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.cache = {}  # Optional: cache embeddings by id

    async def select_top_k_summaries(
        self,
        summaries: List[Dict],
        question: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Select top-K summaries most relevant to the question using cosine similarity of embeddings.
        """
        question_embedding = await self._get_embedding(text=question)

        scored = []
        for summary in summaries:
            summary_id = summary.get("id")
            text = self._summary_to_text(summary)
            embedding = await self._get_embedding(text, cache_key=summary_id)
            similarity = cosine_similarity(
                np.array(question_embedding).reshape(1, -1),
                np.array(embedding).reshape(1, -1)
            )[0][0]
            scored.append((similarity, summary))

        top_k = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        return [s[1] for s in top_k]

    def _summary_to_text(self, summary: Dict) -> str:
        """
        Convert a summary dict to plain text for embedding.
        """
        parts = [summary.get("title", ""), summary.get("summary", "")]
        findings = summary.get("findings", [])
        if isinstance(findings, list):
            parts.extend([f.get("summary", "") for f in findings if isinstance(f, dict)])
        return "\n".join(parts)

    async def _get_embedding(self, text: str, cache_key: str = None) -> List[float]:
        """
        Get embedding for a given text using the LLM client.
        Optionally caches results to avoid recomputation.
        """
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        embedding = await self.llm_client.embed(text)
        if cache_key:
            self.cache[cache_key] = embedding
        return embedding