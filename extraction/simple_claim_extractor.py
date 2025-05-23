"""
Simplified claim extractor for GraphRAG with Deep Seek models.
"""
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SimpleClaimExtractor")

# Import project modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GraphRAGConfig
from models.claim import Claim
from extraction.text_chunker import TextChunk
from utils.llm_client import LLMClient
from utils.async_utils import process_batch_async

class SimpleClaimExtractor:
    """
    A simpler claim extractor that uses JSON output from LLM instead of regex parsing.
    This is more robust with models like Deep Seek that may not follow exact formats.
    """
    
    def __init__(self, llm_client: LLMClient, config: GraphRAGConfig):
        """
        Initialize the simple claim extractor.
        
        Args:
            llm_client: The LLM client
            config: The GraphRAG configuration
        """
        self.llm_client = llm_client
        self.config = config
        
        logger.info(f"Simple claim extractor initialized")
    
    async def extract_from_chunk(
        self, 
        text_chunk: TextChunk,
        entity_names: List[str]
    ) -> List[Claim]:
        """
        Extract claims from a text chunk using JSON extraction.
        
        Args:
            text_chunk: The text chunk to extract from
            entity_names: List of entity names to extract claims for
            
        Returns:
            List of extracted claims
        """
        # Skip if no entities in this chunk
        if not entity_names:
            return []
        
        logger.info(f"Extracting claims from chunk {text_chunk.chunk_id} for {len(entity_names)} entities")
        
        try:
            # Create a JSON extraction prompt
            prompt = self._create_claim_prompt(text_chunk.text, entity_names)
            
            # Extract using JSON format
            extraction_result = await self.llm_client.extract_json(prompt)
            
            # Check if extraction_result is a dictionary
            if not isinstance(extraction_result, dict):
                logger.warning(f"Claim extraction result is not a dictionary: {extraction_result}")
                extraction_result = {"claims": []}
            
            # Parse the result
            claims = self._parse_json_claims(
                extraction_result, text_chunk.source_id, text_chunk.chunk_id
            )
            
            # If no claims found, try fallback
            if not claims:
                logger.info(f"No claims found in JSON extraction, trying fallback")
                claims = await self._extract_simple_fallback(text_chunk, entity_names)
            
            logger.info(f"Extracted {len(claims)} claims from chunk {text_chunk.chunk_id}")
            return claims
            
        except Exception as e:
            logger.error(f"Error in claim extraction from chunk {text_chunk.chunk_id}: {e}")
            
            # Try simple fallback
            return await self._extract_simple_fallback(text_chunk, entity_names)
    
    async def extract_from_chunks(
        self, 
        text_chunks: List[TextChunk],
        entity_map: Dict[str, List[str]]  # Map of chunk_id to entity names in that chunk
    ) -> List[Claim]:
        """
        Extract claims from multiple text chunks.
        
        Args:
            text_chunks: The text chunks to extract from
            entity_map: Map of chunk_id to entity names in that chunk
            
        Returns:
            List of extracted claims
        """
        logger.info(f"Extracting claims from {len(text_chunks)} chunks")
        
        # Process sequentially to avoid overwhelming the model
        all_claims = []
        
        for chunk in text_chunks:
            entity_names = entity_map.get(chunk.chunk_id, [])
            if entity_names:
                try:
                    claims = await self.extract_from_chunk(chunk, entity_names)
                    all_claims.extend(claims)
                    logger.info(f"Processed chunk {chunk.chunk_id}: {len(claims)} claims")
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
        
        logger.info(f"Total: {len(all_claims)} claims extracted")
        return all_claims
    
    def _create_claim_prompt(self, text: str, entity_names: List[str]) -> str:
        """
        Create a simplified JSON claim extraction prompt.
        
        Args:
            text: The text to extract from
            entity_names: List of entity names to extract claims for
            
        Returns:
            The extraction prompt
        """
        entities_str = ", ".join(entity_names)
        
        prompt = f"""
Please analyze the following text and extract factual claims about these entities: {entities_str}

A factual claim is a statement that explicitly presents a verifiable fact about an entity.

Text to analyze:
{text}

Return your results in the following JSON format:
{{
  "claims": [
    {{
      "content": "The factual claim as a concise statement",
      "entities": ["Entity1", "Entity2"]  // List of entities this claim is about
    }}
  ]
}}

Only return the JSON object, no other text.
"""
        return prompt
    
    def _parse_json_claims(
        self,
        extraction_result: Dict[str, Any],
        source_id: str,
        chunk_id: str
    ) -> List[Claim]:
        """
        Parse the JSON claim extraction result.
        
        Args:
            extraction_result: The JSON extraction result
            source_id: The source document ID
            chunk_id: The text chunk ID
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        for claim_data in extraction_result.get("claims", []):
            if isinstance(claim_data, dict) and "content" in claim_data and "entities" in claim_data:
                # Ensure entities is a list
                entities = claim_data["entities"]
                if not isinstance(entities, list):
                    if isinstance(entities, str):
                        entities = [entities]
                    else:
                        entities = []
                
                claim = Claim(
                    content=claim_data["content"],
                    entity_ids=entities  # Temporarily use names as IDs
                )
                claim.add_instance(source_id, chunk_id)
                claims.append(claim)
        
        return claims
    
    async def _extract_simple_fallback(
        self, 
        text_chunk: TextChunk,
        entity_names: List[str]
    ) -> List[Claim]:
        """
        Fallback to an even simpler extraction method if JSON fails.
        
        Args:
            text_chunk: The text chunk to extract from
            entity_names: List of entity names to extract claims for
            
        Returns:
            List of extracted claims
        """
        logger.info(f"Using simple fallback claim extraction for chunk {text_chunk.chunk_id}")
        
        if not entity_names:
            return []
        
        # Create a very simple prompt asking for claims about entities
        prompt = f"""
Please identify factual claims about these entities in the text: {', '.join(entity_names)}
Format each claim as: "CLAIM: [The claim text] - ENTITIES: [Entity1, Entity2]"

Text:
{text_chunk.text}

Factual claims:
"""
        
        try:
            response = await self.llm_client.generate(prompt, temperature=0.1, max_tokens=500)
            
            # Parse the simple format
            claims = []
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if not line or not line.startswith("CLAIM:"):
                    continue
                
                # Try to match "CLAIM: [text] - ENTITIES: [Entity1, Entity2]" pattern
                match = re.search(r"CLAIM:\s*(.+?)\s*-\s*ENTITIES:\s*\[(.*?)\]", line)
                if match:
                    claim_content = match.group(1).strip()
                    entities_str = match.group(2).strip()
                    
                    # Parse the entities
                    claim_entities = [e.strip() for e in entities_str.split(',')]
                    
                    claim = Claim(
                        content=claim_content,
                        entity_ids=claim_entities
                    )
                    claim.add_instance(text_chunk.source_id, text_chunk.chunk_id)
                    claims.append(claim)
            
            logger.info(f"Fallback claim extraction found {len(claims)} claims")
            return claims
            
        except Exception as e:
            logger.error(f"Fallback claim extraction failed: {e}")
            return []