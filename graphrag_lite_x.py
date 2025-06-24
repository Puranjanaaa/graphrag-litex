#!/usr/bin/env python3

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, List, Any

from dotenv import load_dotenv


load_dotenv()  # Load variables from .env file
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleGraphRAG")

# Get the absolute path of the directory containing this script
current_dir = os.path.abspath(os.path.dirname(__file__))

# Add current directory to Python path so all imports work
sys.path.insert(0, current_dir)

# Import the original configuration
from config import GraphRAGConfig
from utils.llm_client import LLMClient
from extraction.text_chunker import TextChunker
from extraction.simple_entity_extractor import SimpleEntityExtractor
from extraction.simple_claim_extractor import SimpleClaimExtractor
from indexing.simple_graph_builder import SimpleGraphBuilder
from indexing.community_detection import CommunityDetector
from indexing.summarizer import CommunitySummarizer
from querying.answer_generator import AnswerGenerator
from main import GraphRAG

class SimpleGraphRAG(GraphRAG):
    """
    A simplified GraphRAG implementation that works better with Deep Seek models.
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the simplified GraphRAG.
        
        Args:
            config: GraphRAG configuration
        """
        logger.info("Initializing Simple GraphRAG")
        
        # Initialize base components
        self.config = config
        self.llm_client = LLMClient(config)
        
        # Initialize text chunker
        self.text_chunker = TextChunker(
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap
        )
        
        # Initialize simplified entity and claim extractors
        self.entity_extractor = SimpleEntityExtractor(self.llm_client, config)
        self.claim_extractor = SimpleClaimExtractor(self.llm_client, config)
        
        # Initialize simplified graph builder
        self.graph_builder = SimpleGraphBuilder(
            self.entity_extractor, 
            self.claim_extractor, 
            config
        )
        
        # Initialize community detection and summarization (use original)
        self.community_detector = CommunityDetector(config)
        self.community_summarizer = CommunitySummarizer(self.llm_client, config)
        
        # Initialize answer generator (use original)
        self.answer_generator = AnswerGenerator(self.llm_client, config)
        
        # Knowledge graph and community hierarchy
        self.knowledge_graph = None
        self.community_hierarchy = None
        
        logger.info("Simple GraphRAG initialized")

    # The rest of the methods are inherited from GraphRAG
async def run_simple_graphrag(
    documents: Dict[str, str],
    questions: List[str],
    lm_studio_url: str = LM_STUDIO_URL,
    save_directory: str = None,
    community_level: str = "C0"
) -> List[str]:
    """
    Run the simplified GraphRAG.

    Args:
        documents: Dictionary of document IDs to text content
        questions: List of questions to answer
        lm_studio_url: URL for the LM Studio API
        save_directory: Directory to save the GraphRAG state to
        community_level: Community level to use for answering questions

    Returns:
        List of generated answers (one per question)
    """
    # Create a configuration for Deep Seek models
    config = GraphRAGConfig(
        # LM Studio settings
        lm_studio_base_url=lm_studio_url,
        llm_temperature=0.1,  # Lower temperature for more deterministic outputs
        llm_max_tokens=2048,
        llm_timeout=180,  # Longer timeout for local inference

        # Chunking settings
        chunk_size=500,
        chunk_overlap=50,

        # Processing settings
        max_self_reflection_iterations=0,  # Disable self-reflection
        context_window_size=4096,
        max_community_answers=3
    )

    # Initialize the simplified GraphRAG
    logger.info("Creating Simple GraphRAG instance")
    graph_rag = SimpleGraphRAG(config)

    # Index the documents
    print("\n=== Indexing Documents with Simple GraphRAG ===\n")
    logger.info(f"Starting indexing of {len(documents)} documents")

    try:
        await graph_rag.index_documents(documents)
        logger.info("Indexing completed successfully")
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise

    # Save the state if requested
    if save_directory:
        logger.info(f"Saving graph to {save_directory}")
        graph_rag.save(save_directory)

    # Answer each question
    print("\n=== Answering Questions with Simple GraphRAG ===\n")
    answers = []
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        print("-" * 80)

        try:
            logger.info(f"Generating answer for question: {question}")
            answer = await graph_rag.answer_question(question, community_level)

            print("\nAnswer:")
            print(answer)
            print("=" * 80)

            answers.append(answer)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            print(f"\nError generating answer: {e}")
            print("=" * 80)
            answers.append(f"[ERROR: {e}]")

    return answers


def read_text_files(directory: str) -> Dict[str, str]:
    """
    Read all text files in a directory.
    
    Args:
        directory: Directory containing text files
        
    Returns:
        Dictionary of {filename: file_content}
    """
    logger.info(f"Reading files from directory: {directory}")
    documents = {}
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith((".txt", ".md", ".py", ".java", ".js", ".html", ".css")):
                file_path = os.path.join(directory, filename)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents[filename] = content
                        logger.info(f"Read file: {filename} ({len(content)} characters)")
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")
    except Exception as e:
        logger.error(f"Error reading directory: {e}")
    
    return documents

async def main():
    parser = argparse.ArgumentParser(description="Simple GraphRAG for Deep Seek models")
    parser.add_argument("--documents", type=str, required=True, help="Directory containing text documents")
    parser.add_argument("--save", type=str, help="Directory to save GraphRAG state")
    parser.add_argument("--level", type=str, default="C0", help="Community level to use (C0, C1, C2, C3)")
    parser.add_argument("--questions", type=str, nargs="+", help="Questions to answer")
    parser.add_argument("--lm-studio-url", type=str, default=LM_STUDIO_URL, 
    # parser.add_argument("--lm-studio-url", type=str, default="http://172.16.6.162:1234/v1", 
                        help="URL for the LM Studio API")
    
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    
    # Load documents
    documents = read_text_files(args.documents)
    
    if not documents:
        logger.error(f"No text files found in {args.documents}")
        print(f"No text files found in {args.documents}")
        return
    
    logger.info(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(documents)} documents.")
    
    # Default questions if none provided
    questions = args.questions or [
        "What are the main topics discussed in these documents?",
        # "What are the key entities mentioned and how are they related?",
        # "What are the most important insights from these documents?"
    ]
    
    # Run GraphRAG
    await run_simple_graphrag(
        documents=documents,
        questions=questions,
        lm_studio_url=args.lm_studio_url,
        save_directory=args.save,
        community_level=args.level
    )

if __name__ == "__main__":
    try:
        # Check if LM Studio is likely running
        import urllib.request
        import urllib.error
        
        lm_studio_url = LM_STUDIO_URL
        try:
            # Just check if the server is responding
            urllib.request.urlopen(f"{lm_studio_url}/models").close()
            print(f"LM Studio server detected at {lm_studio_url}")
        except (urllib.error.URLError, ConnectionRefusedError):
            print(f"Warning: LM Studio server not detected at {lm_studio_url}")
            print("Please make sure LM Studio is running with the API server enabled.")
            print("Continue anyway? (y/n)")
            
            response = input().lower()
            if response != 'y':
                print("Exiting. Please start LM Studio server and try again.")
                sys.exit(1)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted. Exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")