from sentence_transformers import SentenceTransformer, util
from utils.llm_client import LLMClient
from extraction.text_chunker import TextChunker
from config import GraphRAGConfig
import torch
import logging

logger = logging.getLogger(__name__)

async def run_vector_rag(documents, questions, top_k=5):
    config = GraphRAGConfig()
    chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    llm = LLMClient(config)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Chunk all documents
    all_chunks = []
    chunk_texts = []
    for doc_name, text in documents.items():
        chunks = chunker.chunk_text(text, doc_name)
        all_chunks.extend(chunks)
        chunk_texts.extend([c.text for c in chunks])

    if not chunk_texts:
        logger.error("No text chunks were generated from documents")
        return [""] * len(questions)

    # Step 2: Encode all chunks
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    answers = []
    for question in questions:
        try:
            query_embedding = model.encode(question, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
            current_top_k = min(top_k, len(chunk_texts))
            top_indices = torch.topk(scores, k=current_top_k).indices.tolist()
            top_chunks = [chunk_texts[i] for i in top_indices]
            context = "\n\n".join(top_chunks)

            prompt = f"""Use the context below to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer:"""

            # Change from llm.chat() to the correct method name
            answer = await llm.generate(prompt)  # or whatever the correct method is
            answers.append(answer.strip())
        except Exception as e:
            logger.error(f"Error processing question: {question}. Error: {str(e)}")
            answers.append("")  # Return empty string instead of None

    return answers if answers else [""] * len(questions)  # Ensure we always return a list