"""
Knowledge graph data model for GraphRAG.
"""
import json
import networkx as nx
from typing import Dict, Any, Optional, Tuple, Set
import pandas as pd
from models.entity import Entity
from models.relationship import Relationship
from models.claim import Claim

class KnowledgeGraph:
    """
    Represents the knowledge graph built from extracted entities, relationships, and claims.
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.claims: Dict[str, Claim] = {}
        self.community_summaries: Dict[str, Dict[str, Any]] = {}
        self._graph = None  # Lazy-loaded NetworkX graph
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: The entity to add
            
        Returns:
            The ID of the added entity
        """
        # Check if entity with same name already exists
        for existing_id, existing_entity in self.entities.items():
            if existing_entity.name.lower() == entity.name.lower():
                # Merge the entities
                existing_entity.merge(entity)
                return existing_id
        
        # Add new entity
        self.entities[entity.id] = entity
        self._graph = None  # Reset graph
        return entity.id
    
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add a relationship to the knowledge graph.
        
        Args:
            relationship: The relationship to add
            
        Returns:
            The ID of the added relationship
        """
        # Check if a similar relationship already exists
        for existing_id, existing_rel in self.relationships.items():
            if (existing_rel.source_id == relationship.source_id and 
                existing_rel.target_id == relationship.target_id):
                # Merge the relationships
                existing_rel.merge(relationship)
                return existing_id
        
        # Add new relationship
        self.relationships[relationship.id] = relationship
        self._graph = None  # Reset graph
        return relationship.id
    
    def add_claim(self, claim: Claim) -> str:
        """
        Add a claim to the knowledge graph.
        
        Args:
            claim: The claim to add
            
        Returns:
            The ID of the added claim
        """
        # Check if a similar claim already exists
        for existing_id, existing_claim in self.claims.items():
            if existing_claim.content.lower() == claim.content.lower():
                # Just update the entity IDs
                for entity_id in claim.entity_ids:
                    if entity_id not in existing_claim.entity_ids:
                        existing_claim.entity_ids.append(entity_id)
                # Merge instances
                for instance in claim.instances:
                    if instance not in existing_claim.instances:
                        existing_claim.instances.append(instance)
                return existing_id
        
        # Add new claim
        self.claims[claim.id] = claim
        
        # Add to associated entities
        for entity_id in claim.entity_ids:
            if entity_id in self.entities:
                self.entities[entity_id].add_claim(claim.content)
        
        return claim.id
    
    def get_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph representation of the knowledge graph.
        
        Returns:
            A NetworkX graph with entities as nodes and relationships as edges
        """
        if self._graph is None:
            # Create a new graph
            self._graph = nx.Graph()
            
            # Add entities as nodes
            for entity_id, entity in self.entities.items():
                self._graph.add_node(
                    entity_id, 
                    name=entity.name,
                    type=entity.type,
                    description=entity.description
                )
            
            # Add relationships as edges
            for rel_id, rel in self.relationships.items():
                if rel.source_id in self.entities and rel.target_id in self.entities:
                    self._graph.add_edge(
                        rel.source_id,
                        rel.target_id,
                        id=rel_id,
                        description=rel.description,
                        strength=rel.strength
                    )
        
        return self._graph
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """
        Get an entity by its name.
        
        Args:
            name: The name of the entity to find
            
        Returns:
            The entity if found, None otherwise
        """
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
        return None
    
    def add_community_summary(self, community_id: str, summary: Dict[str, Any]):
        """
        Add a summary for a community.
        
        Args:
            community_id: ID of the community
            summary: Dictionary containing summary information
        """
        self.community_summaries[community_id] = summary
    
    def get_community_summary(self, community_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the summary for a community.
        
        Args:
            community_id: ID of the community
            
        Returns:
            The summary dictionary if found, None otherwise
        """
        return self.community_summaries.get(community_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge graph to a dictionary.
        
        Returns:
            A dictionary representation of the knowledge graph
        """
        return {
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "relationships": {rel_id: rel.to_dict() for rel_id, rel in self.relationships.items()},
            "claims": {claim_id: claim.to_dict() for claim_id, claim in self.claims.items()},
            "community_summaries": self.community_summaries
        }
    
    def save(self, file_path: str):
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'KnowledgeGraph':
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            The loaded knowledge graph
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        # Load entities
        for entity_data in data.get("entities", {}).values():
            kg.entities[entity_data["id"]] = Entity.from_dict(entity_data)
        
        # Load relationships
        for rel_data in data.get("relationships", {}).values():
            kg.relationships[rel_data["id"]] = Relationship.from_dict(rel_data)
        
        # Load claims
        for claim_data in data.get("claims", {}).values():
            kg.claims[claim_data["id"]] = Claim.from_dict(claim_data)
        
        # Load community summaries
        kg.community_summaries = data.get("community_summaries", {})
        
        return kg
    
    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert the knowledge graph to pandas DataFrames.
        
        Returns:
            A tuple of (entities_df, relationships_df, claims_df)
        """
        entities_data = []
        for entity_id, entity in self.entities.items():
            entities_data.append({
                "id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "num_instances": len(entity.instances)
            })
        
        relationships_data = []
        for rel_id, rel in self.relationships.items():
            if rel.source_id in self.entities and rel.target_id in self.entities:
                source = self.entities[rel.source_id]
                target = self.entities[rel.target_id]
                relationships_data.append({
                    "id": rel_id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "source_name": source.name,
                    "target_name": target.name,
                    "description": rel.description,
                    "strength": rel.strength,
                    "num_instances": len(rel.instances)
                })
        
        claims_data = []
        for claim_id, claim in self.claims.items():
            entity_names = []
            for entity_id in claim.entity_ids:
                if entity_id in self.entities:
                    entity_names.append(self.entities[entity_id].name)
            
            claims_data.append({
                "id": claim_id,
                "content": claim.content,
                "entity_ids": claim.entity_ids,
                "entity_names": entity_names,
                "num_instances": len(claim.instances)
            })
        
        entities_df = pd.DataFrame(entities_data)
        relationships_df = pd.DataFrame(relationships_data)
        claims_df = pd.DataFrame(claims_data)
        
        return entities_df, relationships_df, claims_df