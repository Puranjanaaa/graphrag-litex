"""
Configuration settings for the GraphRAG system.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv


load_dotenv()  # Load variables from .env file
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")

@dataclass
class GraphRAGConfig:
    """Configuration settings for GraphRAG."""
    # LM Studio settings for locally hosted models

    lm_studio_base_url: str = LM_STUDIO_URL # Default LM Studio endpoint
    llm_model: str = "local-model"  # Model identifier (often ignored for local models)
    llm_temperature: float = 0.2  # Lower temperature for more focused responses with Deep Seek
    llm_max_tokens: int = 4096
    llm_timeout: int = 240  # seconds, increased for local model inference
    logging_enabled: bool = True
    
    # For backward compatibility
    llm_api_key: Optional[str] = None  # Not needed for local LM Studio deployment
    
    # Text chunking settings
    chunk_size: int = 600
    chunk_overlap: int = 100
    
    # Entity extraction settings
    entity_types: List[str] = field(default_factory=list)
    max_self_reflection_iterations: int = 2  # Reduced for local models
    
    # Graph building settings
    min_relationship_strength: float = 0.5
    
    # Community detection settings
    community_resolution: float = 1.0
    min_community_size: int = 3
    
    # Summarization settings
    context_window_size: int = 4096  # Adjusted for Deep Seek model context window
    summary_max_tokens: int = 1024
    
    # Query processing settings
    max_community_answers: int = 5  # Reduced for efficiency with local models
    answer_max_tokens: int = 2048
    
    def __post_init__(self):
        if not self.entity_types:
            self.entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT", "EVENT"]