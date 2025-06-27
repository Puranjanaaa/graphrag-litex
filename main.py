"""
Main GraphRAG class.
"""
from typing import List, Dict, Any, Optional
import asyncio
import os
import time
import json

from config import GraphRAGConfig
from models.knowledge_graph import KnowledgeGraph
from extraction.simple_entity_extractor import SimpleEntityExtractor
from extraction.simple_claim_extractor import SimpleClaimExtractor
from extraction.text_chunker import  TextChunker
from indexing.simple_graph_builder import SimpleGraphBuilder
from indexing.community_detection import CommunityDetector
from indexing.summarizer import CommunitySummarizer
from querying.answer_generator import AnswerGenerator
from utils.llm_client import LLMClient

class GraphRAG:
    """
    Main class for GraphRAG.
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or GraphRAGConfig()
        self.llm_client = LLMClient(self.config)
        
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        self.entity_extractor = SimpleEntityExtractor(self.llm_client, self.config)
        self.claim_extractor = SimpleClaimExtractor(self.llm_client, self.config)
        self.graph_builder = SimpleGraphBuilder(self.entity_extractor, self.claim_extractor, self.config)
        self.community_detector = CommunityDetector(self.config)
        self.community_summarizer = CommunitySummarizer(self.llm_client, self.config)
        self.answer_generator = AnswerGenerator(self.llm_client, self.config)
        
        self.knowledge_graph = None
        self.community_hierarchy = None
    
    async def index_documents(self, documents: Dict[str, str]) -> KnowledgeGraph:
        start_time = time.time()
        print(f"Starting document indexing process for {len(documents)} documents...")
        
        print("Chunking documents...")
        text_chunks = self.text_chunker.chunk_documents(documents)
        print(f"Created {len(text_chunks)} text chunks.")
        
        print("Building knowledge graph...")
        self.knowledge_graph = await self.graph_builder.build_graph(text_chunks)
        
        entities_count = len(self.knowledge_graph.entities)
        relationships_count = len(self.knowledge_graph.relationships)
        claims_count = len(self.knowledge_graph.claims)
        
        print(f"Knowledge graph built with {entities_count} entities, "
              f"{relationships_count} relationships, and {claims_count} claims.")
        
        print("Detecting communities...")
        self.community_hierarchy = self.community_detector.detect_communities(self.knowledge_graph)
        community_levels = list(self.community_hierarchy.keys())
        print(f"Detected community levels: {', '.join(community_levels)}")
        
        print("Generating community summaries...")
        community_summaries = await self.community_summarizer.summarize_communities(
            self.community_hierarchy, self.knowledge_graph, self.community_detector
        )
        summary_count = len(community_summaries)
        print(f"Generated {summary_count} community summaries.")
        
        end_time = time.time()
        print(f"Indexing completed in {end_time - start_time:.2f} seconds.")
        
        return self.knowledge_graph
    
    async def answer_question(self, question: str, community_level: str = "C0", structured: bool = False) -> str:
        if not self.knowledge_graph:
            return "No knowledge graph available. Please index documents first."
        
        print(f"Generating answer for question: {question}")
        print(f"Using community level: {community_level}")
        
        start_time = time.time()
        
        # ✅ Fixed method name
        answer = await self.answer_generator.answer_question(
            question, self.knowledge_graph.get_community_summaries(community_level)
        )
        
        end_time = time.time()
        print(f"Answer generated in {end_time - start_time:.2f} seconds.")
        
        return answer
    
    def save(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.knowledge_graph:
            kg_path = os.path.join(directory, "knowledge_graph.json")
            self.knowledge_graph.save(kg_path)
            print(f"Knowledge graph saved to {kg_path}")
        
        if self.community_hierarchy:
            hierarchy_path = os.path.join(directory, "community_hierarchy.json")
            with open(hierarchy_path, "w") as f:
                json.dump(self.community_hierarchy, f, indent=2)
            print(f"Community hierarchy saved to {hierarchy_path}")
    
    @classmethod
    async def load(cls, directory: str, config: Optional[GraphRAGConfig] = None) -> 'GraphRAG':
        graph_rag = cls(config)
        
        kg_path = os.path.join(directory, "knowledge_graph.json")
        if os.path.exists(kg_path):
            graph_rag.knowledge_graph = KnowledgeGraph.load(kg_path)
            print(f"Knowledge graph loaded from {kg_path}")
        
        hierarchy_path = os.path.join(directory, "community_hierarchy.json")
        if os.path.exists(hierarchy_path):
            with open(hierarchy_path, "r") as f:
                graph_rag.community_hierarchy = json.load(f)
            print(f"Community hierarchy loaded from {hierarchy_path}")
        
        return graph_rag
