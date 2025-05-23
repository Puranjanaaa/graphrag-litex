"""
Claim data model for GraphRAG.
"""
from dataclasses import dataclass, field
import uuid
from typing import List, Dict, Any

@dataclass
class Claim:
    """
    Represents a factual claim associated with entities in the knowledge graph.
    
    Attributes:
        content: The text content of the claim
        entity_ids: List of entity IDs associated with this claim
        id: Unique identifier for the claim
        instances: List of dictionaries containing source information about claim instances
    """
    content: str
    entity_ids: List[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instances: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_instance(self, source_id: str, text_chunk_id: str):
        """Add a source instance where this claim was identified."""
        self.instances.append({
            "source_id": source_id,
            "text_chunk_id": text_chunk_id
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the claim to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "entity_ids": self.entity_ids,
            "instances": self.instances
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Claim':
        """Create a claim from a dictionary."""
        claim = cls(
            content=data["content"],
            entity_ids=data["entity_ids"],
            id=data.get("id", str(uuid.uuid4()))
        )
        claim.instances = data.get("instances", [])
        return claim