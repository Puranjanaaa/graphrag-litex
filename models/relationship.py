"""
Relationship data model for GraphRAG.
"""
from dataclasses import dataclass, field
import uuid
from typing import List, Dict, Any

@dataclass
class Relationship:
    """
    Represents a relationship edge in the knowledge graph.
    
    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        description: Description of the relationship
        strength: Strength of the relationship (0-1)
        id: Unique identifier for the relationship
        instances: List of dictionaries containing source information about relationship instances
    """
    source_id: str
    target_id: str
    description: str
    strength: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instances: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_instance(self, source_id: str, text_chunk_id: str):
        """Add a source instance where this relationship was identified."""
        self.instances.append({
            "source_id": source_id,
            "text_chunk_id": text_chunk_id
        })
    
    def merge(self, other: 'Relationship') -> 'Relationship':
        """
        Merge another relationship into this one.
        
        Args:
            other: Another relationship to merge
            
        Returns:
            The merged relationship (self)
        """
        # Merge descriptions (take the longer one)
        if len(other.description) > len(self.description):
            self.description = other.description
            
        # Update strength (take the average)
        self.strength = (self.strength + other.strength) / 2
            
        # Merge instances
        for instance in other.instances:
            if instance not in self.instances:
                self.instances.append(instance)
                
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship to a dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "description": self.description,
            "strength": self.strength,
            "instances": self.instances
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create a relationship from a dictionary."""
        relationship = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            description=data["description"],
            strength=data["strength"],
            id=data.get("id", str(uuid.uuid4()))
        )
        relationship.instances = data.get("instances", [])
        return relationship