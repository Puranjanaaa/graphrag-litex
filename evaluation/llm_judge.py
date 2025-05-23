import json
from utils.llm_client import LLMClient
from config import GraphRAGConfig

EVAL_CRITERIA = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects of the question?",
    "diversity": "How varied and rich is the answer in terms of perspectives and insights?",
    "empowerment": "How well does the answer help the reader understand and make informed judgments?",
    "directness": "How specifically and clearly does the answer address the question?"
}

async def judge_answers(question: str, answer1: str, answer2: str) -> dict:
    config = GraphRAGConfig()
    client = LLMClient(config)

    results = {"question": question, "evaluations": []}

    for criterion_name, criterion_def in EVAL_CRITERIA.items():
        prompt = f"""
You are a helpful AI judge. Compare two answers to a question and assess which one is better.

Criterion: {criterion_name.upper()}
Definition: {criterion_def}

Question: {question}

Answer 1:
{answer1}

Answer 2:
{answer2}

Instructions:
1. Choose the better answer: 1, 2, or 0 if it's a tie.
2. Score each answer on a scale of 0-100 for how well it meets the criterion.
3. Give a short justification.

Respond in this JSON format:

{{
  "winner": 1 | 2 | 0,
  "score1": int (0-100),
  "score2": int (0-100),
  "reasoning": "short justification"
}}
"""
        try:
            data = await client.extract_json(prompt)  # <- more robust than .generate + json.loads
        except Exception as e:
            data = {
                "winner": 0,
                "score1": 0,
                "score2": 0,
                "reasoning": f"Invalid or failed response: {str(e)}"
            }

        results["evaluations"].append({
            "criterion": criterion_name,
            "judgment": {
                "winner": data.get("winner", 0),
                "reasoning": data.get("reasoning", ""),
            },
            "score1": data.get("score1", "N/A"),
            "score2": data.get("score2", "N/A")
        })

    return results
