"""
Fixed text chunking utilities for GraphRAG.
"""
from typing import List, Dict, Any
import uuid
import asyncio
import tiktoken
import logging


# Set up logging
logger = logging.getLogger("TextChunker")

class TextChunk:
    """
    Represents a chunk of text from a source document.
    """
    
    def __init__(self, text: str, source_id: str, chunk_id: str = None):
        """
        Initialize a text chunk.
        
        Args:
            text: The text content of the chunk
            source_id: ID of the source document
            chunk_id: ID of the chunk (generated if not provided)
        """
        self.text = text
        self.source_id = source_id
        self.chunk_id = chunk_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the text chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """Create a text chunk from a dictionary."""
        return cls(
            text=data["text"],
            source_id=data["source_id"],
            chunk_id=data.get("chunk_id", str(uuid.uuid4()))
        )

class TextChunker:
    """
    Utility for chunking text documents.
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)  # Ensure overlap isn't too large
        self.encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str, source_id: str) -> List[TextChunk]:
        """
        Blocking function. Split text into chunks.
        Call this via `await asyncio.to_thread(...)` from async code.
        
        Args:
            text: The text to chunk
            source_id: ID of the source document
            
        Returns:
            List of text chunks
        """
        logger.info(f"Chunking text for source_id={source_id}, text length={len(text)} characters")
        
        tokens = self.encoding.encode(text)
        token_count = len(tokens)
        logger.info(f"Token count: {token_count}")
        
        if token_count == 0:
            logger.warning(f"Empty document: {source_id}")
            return []

        if token_count <= self.chunk_size:
            logger.info(f"Document fits in a single chunk: {token_count} ≤ {self.chunk_size}")
            chunk_text = self.encoding.decode(tokens)
            return [TextChunk(text=chunk_text, source_id=source_id, chunk_id=f"{source_id}_0")]
        
        chunks = []
        chunk_start = 0
        chunk_count = 0
        
        while chunk_start < token_count:
            chunk_end = min(chunk_start + self.chunk_size, token_count)
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source_id=source_id,
                    chunk_id=f"{source_id}_{chunk_count}"
                )
            )
            
            chunk_count += 1
            chunk_start = chunk_end - self.chunk_overlap
            
            if chunk_start + self.chunk_size - self.chunk_overlap >= token_count:
                break

        logger.info(f"Created {len(chunks)} chunks for document {source_id}")
        return chunks

    
    async def chunk_documents(self, documents: Dict[str, str]) -> List[TextChunk]:
        """
        Split multiple documents into chunks asynchronously.

        Args:
            documents: Dictionary of {document_id: document_text}
            
        Returns:
            List of text chunks across all documents
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        # Run chunk_text in parallel threads
        results = await asyncio.gather(
            *(asyncio.to_thread(self.chunk_text, doc_text, doc_id) for doc_id, doc_text in documents.items())
        )

        all_chunks = [chunk for doc_chunks in results for chunk in doc_chunks]
        
        logger.info(f"Created {len(all_chunks)} chunks total across all documents")
        return all_chunks
