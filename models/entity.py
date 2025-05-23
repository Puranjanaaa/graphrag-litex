"""
Entity data model for GraphRAG.
"""
from dataclasses import dataclass, field
import uuid
from typing import List, Dict, Any

@dataclass
class Entity:
    """
    Represents an entity node in the knowledge graph.
    
    Attributes:
        name: Name of the entity
        type: Type of entity (e.g., PERSON, ORGANIZATION)
        description: Description of the entity
        id: Unique identifier for the entity
        instances: List of dictionaries containing source information about entity instances
        claims: List of claims related to the entity
    """
    name: str
    type: str
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instances: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    
    def add_instance(self, source_id: str, text_chunk_id: str):
        """Add a source instance where this entity was identified."""
        self.instances.append({
            "source_id": source_id,
            "text_chunk_id": text_chunk_id
        })
    
    def add_claim(self, claim: str):
        """Add a claim about this entity."""
        if claim not in self.claims:
            self.claims.append(claim)
    
    def merge(self, other: 'Entity') -> 'Entity':
        """
        Merge another entity into this one.
        
        Args:
            other: Another entity to merge
            
        Returns:
            The merged entity (self)
        """
        # Merge descriptions
        if len(other.description) > len(self.description):
            self.description = other.description
            
        # Merge instances
        for instance in other.instances:
            if instance not in self.instances:
                self.instances.append(instance)
                
        # Merge claims
        for claim in other.claims:
            if claim not in self.claims:
                self.claims.append(claim)
                
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "instances": self.instances,
            "claims": self.claims
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create an entity from a dictionary."""
        entity = cls(
            name=data["name"],
            type=data["type"],
            description=data["description"],
            id=data.get("id", str(uuid.uuid4()))
        )
        entity.instances = data.get("instances", [])
        entity.claims = data.get("claims", [])
        return entity