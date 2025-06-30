"""
Updated graph builder for Deep Seek models that uses the simplified extractors.
"""
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SimpleGraphBuilder")

# Import project modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from extraction.text_chunker import TextChunk
from extraction.simple_entity_extractor import SimpleEntityExtractor
from extraction.simple_claim_extractor import SimpleClaimExtractor
from extraction.entity_resolver import EntityResolver  # ✅ NEW: Import resolver

class SimpleGraphBuilder:
    """
    Builds a knowledge graph using the simplified extractors.
    """
    
    def __init__(
        self, 
        entity_extractor: SimpleEntityExtractor,
        claim_extractor: SimpleClaimExtractor,
        config: GraphRAGConfig
    ):
        """
        Initialize the graph builder.
        
        Args:
            entity_extractor: The entity extractor
            claim_extractor: The claim extractor
            config: The GraphRAG configuration
        """
        self.entity_extractor = entity_extractor
        self.claim_extractor = claim_extractor
        self.config = config
        self.min_relationship_strength = config.min_relationship_strength
        self.entity_resolver = EntityResolver()  # ✅ NEW: Instantiate resolver
        
        logger.info(f"Simple graph builder initialized")
    
    async def build_graph(self, text_chunks: List[TextChunk]) -> KnowledgeGraph:
        """
        Build a knowledge graph from text chunks.
        
        Args:
            text_chunks: The text chunks to build the graph from
            
        Returns:
            The built knowledge graph
        """
        # Extract entities and relationships
        logger.info(f"Extracting entities and relationships from {len(text_chunks)} chunks")
        entities, relationships = await self.entity_extractor.extract_from_chunks(text_chunks)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")

        logger.info("Resolving semantically equivalent entities...")
        entities = self.entity_resolver.resolve_entities(entities)
        logger.info(f"{len(entities)} canonical entities after resolution")

        # Create a new knowledge graph
        kg = KnowledgeGraph()
        
        # Add entities to the graph
        entity_id_map = {}  # Maps entity names to IDs
        for entity in entities:
            entity_id = kg.add_entity(entity)
            entity_id_map[entity.name] = entity_id
        
        logger.info(f"Added {len(entity_id_map)} entities to the knowledge graph")
        
        # Resolve relationship entity IDs and add to the graph
        added_relationships = 0
        for relationship in relationships:
            # Skip relationships with strength below threshold
            if relationship.strength < self.min_relationship_strength:
                continue
                
            # Get source and target entity IDs
            source_name = relationship.source_id  # Temporarily used name as ID
            target_name = relationship.target_id  # Temporarily used name as ID
            
            source_id = entity_id_map.get(source_name)
            target_id = entity_id_map.get(target_name)
            
            # Skip relationships with unknown entities
            if not source_id or not target_id:
                continue
                
            # Update relationship with actual entity IDs
            relationship.source_id = source_id
            relationship.target_id = target_id
            
            # Add to knowledge graph
            kg.add_relationship(relationship)
            added_relationships += 1
        
        logger.info(f"Added {added_relationships} relationships to the knowledge graph")
        
        # Create entity map for claim extraction
        entity_map = {}  # Map of chunk_id to entity names in that chunk
        for entity in entities:
            for instance in entity.instances:
                chunk_id = instance.get("text_chunk_id")
                if chunk_id:
                    if chunk_id not in entity_map:
                        entity_map[chunk_id] = []
                    if entity.name not in entity_map[chunk_id]:
                        entity_map[chunk_id].append(entity.name)
        
        # Extract claims
        logger.info(f"Extracting claims from {len(text_chunks)} chunks")
        claims = await self.claim_extractor.extract_from_chunks(text_chunks, entity_map)
        
        logger.info(f"Extracted {len(claims)} claims")
        
        # Resolve claim entity IDs and add to the graph
        added_claims = 0
        for claim in claims:
            # Resolve entity IDs
            entity_ids = []
            for entity_name in claim.entity_ids:  # Temporarily used names as IDs
                entity_id = entity_id_map.get(entity_name)
                if entity_id:
                    entity_ids.append(entity_id)
            
            # Skip claims with no known entities
            if not entity_ids:
                continue
                
            # Update claim with actual entity IDs
            claim.entity_ids = entity_ids
            
            # Add to knowledge graph
            kg.add_claim(claim)
            added_claims += 1
        
        logger.info(f"Added {added_claims} claims to the knowledge graph")
        
        # Final knowledge graph stats
        logger.info(f"Final knowledge graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships, {len(kg.claims)} claims")
        
        return kg