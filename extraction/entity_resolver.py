import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from models.entity import Entity


class EntityResolver:
    def __init__(self, threshold: float = 0.85):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.strip()

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        canonical_entities: Dict[str, Entity] = {}
        name_to_id: Dict[str, str] = {}

        normalized_names = [self.normalize(e.name) for e in entities]
        embeddings = self.embed(normalized_names)

        for i, entity in enumerate(entities):
            name = self.normalize(entity.name)
            emb = embeddings[i]

            best_match = None
            best_score = -1

            for canon_name, canon_entity in canonical_entities.items():
                canon_index = list(canonical_entities.keys()).index(canon_name)
                canon_emb = embeddings[canon_index]
                score = np.dot(emb, canon_emb) / (np.linalg.norm(emb) * np.linalg.norm(canon_emb))
                if score > self.threshold and score > best_score:
                    best_match = canon_name
                    best_score = score

            if best_match:
                print(f"[EntityResolver] Merging '{entity.name}' into canonical entity '{best_match}' (score={best_score:.2f})")
                canonical_entities[best_match].merge(entity)
                name_to_id[name] = canonical_entities[best_match].id
            else:
                print(f"[EntityResolver] New canonical entity added: '{entity.name}'")
                canonical_entities[name] = entity
                name_to_id[name] = entity.id

        return list(canonical_entities.values())
