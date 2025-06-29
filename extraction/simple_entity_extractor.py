"""
Simplified entity extractor for GraphRAG with Deep Seek models.
"""
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SimpleEntityExtractor")

# Import project modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GraphRAGConfig
from models.entity import Entity
from models.relationship import Relationship
from extraction.text_chunker import TextChunk
from utils.llm_client import LLMClient
from utils.async_utils import process_batch_async

class SimpleEntityExtractor:
    """
    A simpler entity extractor that uses JSON output from LLM instead of regex parsing.
    This is more robust with models like Deep Seek that may not follow exact formats.
    """
    
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        """
        Initialize the simple entity extractor.
        
        Args:
            llm_client: The LLM client
            config: The GraphRAG configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.entity_types = config.entity_types
        self.max_self_reflection_iterations = 0  # Disable for simplicity
        
        logger.info(f"Simple entity extractor initialized with entity types: {self.entity_types}")
    
    async def extract_from_chunk(
        self, 
        text_chunk: TextChunk
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a text chunk using JSON extraction.
        
        Args:
            text_chunk: The text chunk to extract from
            
        Returns:
            A tuple of (entities, relationships)
        """
        logger.info(f"Extracting from chunk: {text_chunk.chunk_id}")
        
        # Create a simplified prompt for Deep Seek
        prompt = self._create_extraction_prompt(text_chunk.text)
        
        try:
            # Extract using JSON format
            extraction_result = await self.llm_client.extract_json(prompt)
            
            # Check if extraction_result is a dictionary
            if not isinstance(extraction_result, dict):
                logger.warning(f"Extraction result is not a dictionary: {extraction_result}")
                extraction_result = {"entities": [], "relationships": []}
            
            # Parse the result
            entities, relationships = self._parse_json_extraction(
                extraction_result, text_chunk.source_id, text_chunk.chunk_id
            )
            
            # If no entities found, try fallback
            if not entities:
                logger.info(f"No entities found in JSON extraction, trying fallback")
                entities, relationships = await self._extract_simple_fallback(text_chunk)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {text_chunk.chunk_id}")
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error in extraction from chunk {text_chunk.chunk_id}: {e}")
            # Try a simpler approach as fallback
            return await self._extract_simple_fallback(text_chunk)
    
    async def extract_from_chunks(
        self, 
        text_chunks: List[TextChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from multiple text chunks concurrently.
        
        Args:
            text_chunks: The text chunks to extract from
            
        Returns:
            A tuple of (entities, relationships)
        """
        logger.info(f"Extracting from {len(text_chunks)} chunks")

        # Launch concurrent extraction tasks
        tasks = [self.extract_from_chunk(chunk) for chunk in text_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_entities = []
        all_relationships = []

        for chunk, result in zip(text_chunks, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {chunk.chunk_id}: {result}")
                continue
            entities, relationships = result
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            logger.info(f"Processed chunk {chunk.chunk_id}: {len(entities)} entities, {len(relationships)} relationships")

        logger.info(f"Total: {len(all_entities)} entities, {len(all_relationships)} relationships")
        return all_entities, all_relationships

    
    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create a simplified JSON extraction prompt for Deep Seek.
        
        Args:
            text: The text to extract from
            
        Returns:
            The extraction prompt
        """
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""
Please analyze the following text and extract:
1. Entities of these types: {entity_types_str}
2. Relationships between these entities

Text to analyze:
{text}

Return your results in the following JSON format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity Type",
      "description": "Brief description of the entity"
    }}
  ],
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "description": "Description of relationship",
      "strength": 0.8  // Number between 0 and 1 indicating relationship strength
    }}
  ]
}}

Only return the JSON object, no other text.
"""
        return prompt
    
    def _parse_json_extraction(
        self,
        extraction_result: Dict[str, Any],
        source_id: str,
        chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Parse the JSON extraction result.
        
        Args:
            extraction_result: The JSON extraction result
            source_id: The source document ID
            chunk_id: The text chunk ID
            
        Returns:
            A tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        
        # Extract entities
        for entity_data in extraction_result.get("entities", []):
            if isinstance(entity_data, dict) and "name" in entity_data and "type" in entity_data:
                entity = Entity(
                    name=entity_data["name"],
                    type=entity_data["type"],
                    description=entity_data.get("description", f"A {entity_data['type']}")
                )
                entity.add_instance(source_id, chunk_id)
                entities.append(entity)
        
        # Extract relationships
        for rel_data in extraction_result.get("relationships", []):
            if isinstance(rel_data, dict) and "source" in rel_data and "target" in rel_data:
                # Find the corresponding entity objects
                source_entity = next((e for e in entities if e.name == rel_data["source"]), None)
                target_entity = next((e for e in entities if e.name == rel_data["target"]), None)
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        source_id=source_entity.name,  # Temporarily use names as IDs
                        target_id=target_entity.name,  # Temporarily use names as IDs
                        description=rel_data.get("description", "Related to"),
                        strength=rel_data.get("strength", 0.5)
                    )
                    relationship.add_instance(source_id, chunk_id)
                    relationships.append(relationship)
        
        return entities, relationships
    
    async def _extract_simple_fallback(
        self, 
        text_chunk: TextChunk
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Fallback to an even simpler extraction method if JSON fails.
        
        Args:
            text_chunk: The text chunk to extract from
            
        Returns:
            A tuple of (entities, relationships)
        """
        logger.info(f"Using simple fallback extraction for chunk {text_chunk.chunk_id}")
        
        # Create a very simple prompt asking for just entity names and types
        prompt = f"""
Please identify the key entities in this text. Only include entities of these types: {', '.join(self.entity_types)}.
Format: "Entity Name (Entity Type)"

Text:
{text_chunk.text}

Key entities:
"""
        
        try:
            response = await self.llm_client.generate(prompt, temperature=0.1, max_tokens=500)
            
            # Parse the simple format
            entities = []
            relationships = []
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to match "Entity Name (Entity Type)" pattern
                match = re.search(r"(.+?)\s*\(([^)]+)\)", line)
                if match:
                    entity_name = match.group(1).strip()
                    entity_type = match.group(2).strip()
                    
                    entity = Entity(
                        name=entity_name,
                        type=entity_type,
                        description=f"A {entity_type}"
                    )
                    entity.add_instance(text_chunk.source_id, text_chunk.chunk_id)
                    entities.append(entity)
            
            logger.info(f"Fallback extraction found {len(entities)} entities")
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return [], []