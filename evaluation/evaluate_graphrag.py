import json
import asyncio
import csv
import os
import logging
import time
import traceback
from datetime import datetime

# Disable urllib3 connection pool logs
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
# Disable other verbose loggers from Hugging Face libraries
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from evaluation.llm_judge import judge_answers
from baselines.vector_rag import run_vector_rag
from utils.io_utils import read_text_files
from graphrag_lite_x import run_simple_graphrag

# Disable INFO level logs from LLMClient to prevent prompt printing
logging.getLogger("LLMClient").setLevel(logging.WARNING)

# Configure logging for production deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] GraphRAG: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("GraphRAG-Evaluation")

# Sample corpus and questions
DOC_DIR = "./data/documents"
QUESTIONS = [
    "What are the main themes discussed in this corpus?",
    "What key entities are involved and how do they relate?",
    "What insights can be derived about the topic's evolution over time?"
]

async def evaluate():
    evaluation_id = f"eval-{int(time.time())}"
    start_time = time.time()
    logger.info(f"[{evaluation_id}] Starting evaluation pipeline")
    
    try:
        # Load documents
        load_start = time.time()
        documents = read_text_files(DOC_DIR)
        load_time = time.time() - load_start
        logger.info(f"[{evaluation_id}] Loaded {len(documents)} documents from {DOC_DIR} in {load_time:.2f}s")
        logger.info(f"[{evaluation_id}] Total document content: {sum(len(d) for d in documents.values())/1024:.2f} KB")

        # Run SimpleGraphRAG
        logger.info(f"[{evaluation_id}] Running SimpleGraphRAG...")
        graphrag_start = time.time()
        # Call run_simple_graphrag and capture the result as a tuple (answers, stats) if available
        result = await run_simple_graphrag(
            documents=documents,
            questions=QUESTIONS,
            save_directory=None,
            community_level="C0"
        )
        
        # Check if result is a tuple (indicating answers and stats were returned)
        if isinstance(result, tuple) and len(result) == 2:
            simple_answers, graph_stats = result
            has_graph_stats = True
        else:
            # Just answers were returned (no stats)
            simple_answers = result
            has_graph_stats = False
            
        graphrag_time = time.time() - graphrag_start
        
        if simple_answers is None:
            raise RuntimeError("SimpleGraphRAG returned None. Check model response or implementation.")
            
        logger.info(f"[{evaluation_id}] SimpleGraphRAG completed in {graphrag_time:.2f}s")
        logger.info(f"[{evaluation_id}] Generated {len(simple_answers)} answers")
        
        # Display document and corpus statistics as an alternative to graph stats
        logger.info(f"[{evaluation_id}] Document and Processing Statistics:")
        logger.info(f"[{evaluation_id}] - Document count: {len(documents)}")
        logger.info(f"[{evaluation_id}] - Average document size: {sum(len(d) for d in documents.values())/len(documents)/1024:.2f} KB")
        logger.info(f"[{evaluation_id}] - Total content processed: {sum(len(d) for d in documents.values())/1024:.2f} KB")
        logger.info(f"[{evaluation_id}] - Processing speed: {sum(len(d) for d in documents.values())/1024/graphrag_time:.2f} KB/s")
        
        # Display stats if they're available from the function return value
        if has_graph_stats and isinstance(graph_stats, dict):
            logger.info(f"[{evaluation_id}] Knowledge Graph Statistics:")
            for key, value in graph_stats.items():
                if isinstance(value, float):
                    logger.info(f"[{evaluation_id}] - {key}: {value:.4f}")
                elif isinstance(value, int):
                    logger.info(f"[{evaluation_id}] - {key}: {value:,}")
                elif isinstance(value, dict):
                    logger.info(f"[{evaluation_id}] - {key}: {len(value)} items")
                    # If it's a distribution, show top items
                    if len(value) > 0:
                        top_items = sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]
                        logger.info(f"[{evaluation_id}]   Top 5: {top_items}")
                else:
                    logger.info(f"[{evaluation_id}] - {key}: {value}")

        # Run Vector RAG baseline
        logger.info(f"[{evaluation_id}] Running Vector RAG baseline...")
        vector_start = time.time()
        baseline_answers = await run_vector_rag(
            documents=documents,
            questions=QUESTIONS
        )
        vector_time = time.time() - vector_start
        
        if baseline_answers is None:
            raise RuntimeError("VectorRAG baseline returned None. Check model response or implementation.")
            
        logger.info(f"[{evaluation_id}] VectorRAG completed in {vector_time:.2f}s")
        logger.info(f"[{evaluation_id}] Generated {len(baseline_answers)} answers")

        # Judge answers
        logger.info(f"[{evaluation_id}] Evaluating system performances...")
        judge_start = time.time()
        systems = ["SimpleGraphRAG", "VectorRAG"]
        evaluations = []

        for q_idx, (q, simple_ans, baseline_ans) in enumerate(zip(QUESTIONS, simple_answers, baseline_answers)):
            logger.info(f"[{evaluation_id}] Judging question {q_idx+1}/{len(QUESTIONS)}")
            eval_result = await judge_answers(q, simple_ans, baseline_ans)
            for entry in eval_result["evaluations"]:
                winner = entry["judgment"].get("winner", 0)
                entry.update({
                    "question": q,
                    "system1": systems[0],
                    "system2": systems[1],
                    "winning_system": systems[winner - 1] if winner in [1, 2] else "Tie",
                    "score1": entry.get("score1", "N/A"),
                    "score2": entry.get("score2", "N/A")
                })
                evaluations.append(entry)
        

        print("\n=== Evaluation Results ===\n")
        judge_time = time.time() - judge_start
        logger.info(f"[{evaluation_id}] Evaluation completed in {judge_time:.2f}s")

        # Summarize evaluation results
        wins = {"SimpleGraphRAG": 0, "VectorRAG": 0, "Tie": 0}
        for entry in evaluations:
            wins[entry["winning_system"]] += 1
            
        total_comparisons = len(evaluations)
        logger.info(f"[{evaluation_id}] Evaluation Summary:")
        logger.info(f"[{evaluation_id}] - Total comparisons: {total_comparisons}")
        logger.info(f"[{evaluation_id}] - SimpleGraphRAG wins: {wins['SimpleGraphRAG']} ({wins['SimpleGraphRAG']/total_comparisons*100:.1f}%)")
        logger.info(f"[{evaluation_id}] - VectorRAG wins: {wins['VectorRAG']} ({wins['VectorRAG']/total_comparisons*100:.1f}%)")
        logger.info(f"[{evaluation_id}] - Ties: {wins['Tie']} ({wins['Tie']/total_comparisons*100:.1f}%)")

        # Save results
        save_evaluation_to_csv(evaluations, f"evaluation/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Performance metrics
        total_time = time.time() - start_time
        logger.info(f"[{evaluation_id}] Performance Metrics:")
        logger.info(f"[{evaluation_id}] - Total evaluation time: {total_time:.2f}s")
        logger.info(f"[{evaluation_id}] - Document loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        logger.info(f"[{evaluation_id}] - SimpleGraphRAG: {graphrag_time:.2f}s ({graphrag_time/total_time*100:.1f}%)")
        logger.info(f"[{evaluation_id}] - VectorRAG: {vector_time:.2f}s ({vector_time/total_time*100:.1f}%)")
        logger.info(f"[{evaluation_id}] - Judging: {judge_time:.2f}s ({judge_time/total_time*100:.1f}%)")
        
        logger.info(f"[{evaluation_id}] Evaluation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"[{evaluation_id}] Critical error in evaluation pipeline: {str(e)}")
        logger.error(f"[{evaluation_id}] Traceback: {traceback.format_exc()}")
        raise


def save_evaluation_to_csv(evaluations, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "question", "criterion", "winner", "winning_system",
                "system1", "score1", "system2", "score2", "reasoning"
            ])

            for row in evaluations:
                writer.writerow([
                    row["question"],
                    row["criterion"],
                    row["judgment"].get("winner", "N/A"),
                    "GraphRAGLiteX" if row["winning_system"] == "SimpleGraphRAG" else row["winning_system"],
                    "GraphRAGLiteX" if row["system1"] == "SimpleGraphRAG" else row["system1"],
                    row["score1"],
                    "GraphRAGLiteX" if row["system2"] == "SimpleGraphRAG" else row["system2"],
                    row["score2"],
                    row["judgment"].get("reasoning", "").replace("\n", " ").strip()
                ])

        logger.info(f"Saved evaluation results to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")



if __name__ == "__main__":
    try:
        asyncio.run(evaluate())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        logger.critical(traceback.format_exc())
