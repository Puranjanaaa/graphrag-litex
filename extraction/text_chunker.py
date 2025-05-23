"""
Fixed text chunking utilities for GraphRAG.
"""
from typing import List, Dict, Any
import uuid
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
        Split text into chunks.
        
        Args:
            text: The text to chunk
            source_id: ID of the source document
            
        Returns:
            List of text chunks
        """
        # Convert text to tokens
        logger.info(f"Chunking text for source_id={source_id}, text length={len(text)} characters")
        tokens = self.encoding.encode(text)
        token_count = len(tokens)
        logger.info(f"Token count: {token_count}")
        
        # Handle empty or very small documents
        if token_count == 0:
            logger.warning(f"Empty document: {source_id}")
            return []
        
        if token_count <= self.chunk_size:
            # Document fits in a single chunk
            logger.info(f"Document fits in a single chunk: {token_count} ≤ {self.chunk_size}")
            chunk_text = self.encoding.decode(tokens)
            chunk = TextChunk(
                text=chunk_text,
                source_id=source_id,
                chunk_id=f"{source_id}_0"
            )
            return [chunk]
        
        # Split tokens into chunks
        chunks = []
        chunk_start = 0
        chunk_count = 0
        
        while chunk_start < token_count:
            # Determine the end of the current chunk
            chunk_end = min(chunk_start + self.chunk_size, token_count)
            
            logger.debug(f"Creating chunk {chunk_count} from positions {chunk_start} to {chunk_end}")
            
            # Convert chunk tokens back to text
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create a text chunk
            chunk = TextChunk(
                text=chunk_text,
                source_id=source_id,
                chunk_id=f"{source_id}_{chunk_count}"
            )
            chunks.append(chunk)
            
            # Advance the counter
            chunk_count += 1
            
            # Move the start index for the next chunk (with overlap)
            chunk_start = chunk_end - self.chunk_overlap
            
            # If the next chunk would be too small, just end here
            if chunk_start + self.chunk_size - self.chunk_overlap >= token_count:
                break
        
        logger.info(f"Created {len(chunks)} chunks for document {source_id}")
        return chunks
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[TextChunk]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: Dictionary of {document_id: document_text}
            
        Returns:
            List of text chunks across all documents
        """
        logger.info(f"Chunking {len(documents)} documents")
        all_chunks = []
        
        for doc_id, doc_text in documents.items():
            logger.info(f"Processing document {doc_id}, length={len(doc_text)} characters")
            doc_chunks = self.chunk_text(doc_text, doc_id)
            all_chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks total across all documents")
        return all_chunks