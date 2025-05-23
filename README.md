# 📘 GraphRAG-LiteX: Lightweight Graph-Based Retrieval-Augmented Generation

![GraphRAG-LiteX Banner](https://user-images.githubusercontent.com/12345678/graphrag_litex_banner.png)

GraphRAG-LiteX is a **lightweight**, **modular**, and **locally executable** version of the GraphRAG framework, built to run on modern LLMs like DeepSeek via LM Studio. It enables **knowledge graph-based retrieval-augmented generation** and provides a built-in evaluation framework comparing GraphRAG against traditional vector-based RAG pipelines.

---

## 🌟 Key Features

- 🧠 Graph-based RAG: Combines claims and entity extraction into a dynamic knowledge graph.
- 🔄 Async LLM integration: Optimized for high-throughput local processing with LM Studio.
- 🧪 Evaluation-ready: Includes a comprehensive side-by-side evaluation with VectorRAG.
- ⚙️ Modular: Easily extend components like extraction prompts or graph logic.
- 💡 Insightful: Evaluates outputs based on global coherence, faithfulness, and empowerment.

---

## 📂 Project Structure
.
├── config.py
├── graphrag_lite_x.py (Main pipeline - graph builder + answer generator)
├── main.py (Optional entry point)
├── requirements.txt
├── .env (API keys and model info)
├── README.md
│
├── utils/
│ ├── async_utils.py
│ ├── io_utils.py
│ ├── prompts.py
│ └── llm_client.py
│
├── models/
│ ├── claim.py
│ ├── entity.py
│ ├── relationship.py
│ └── knowledge_graph.py
│
└── evaluation/
├── evaluate_graphrag.py (Compares GraphRAG-LiteX vs VectorRAG)
├── llm_judge.py (LLM-based evaluation)
└── *.csv (Saved evaluation results)


---

## 🧰 Requirements

- Python 3.9+
- LM Studio (for local LLM inference)
- DeepSeek-compatible model (e.g. `deepseek-r1-distill-qwen-7b`)

### 🔧 Setup

```bash
pip install -r requirements.txt
Add a .env file:

LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=deepseek-r1-distill-qwen-7b
```
### 🚀 Running the Pipeline

Run GraphRAG-LiteX:
```bash
python3 graphrag_lite_x.py --documents ./data/documents
```

Evaluate GraphRAG-LiteX vs VectorRAG:
```bash
python3 -m evaluation.evaluate_graphrag
```

### 🖼️ System Architecture

Documents → Claim & Entity Extraction
Nodes/Edges → Knowledge Graph
Query → Subgraph Retrieval
Subgraph → LLM → Final Answer

### 📊 Sample Evaluation Output

System	Wins	Percentage
GraphRAG-LiteX	10	83.3%
VectorRAG	2	16.7%

📈 Based on criteria like comprehensiveness, faithfulness, empowerment, and coherence.

### 🧠 Model Recommendation

Tested and tuned with:

✅ deepseek-r1-distill-qwen-7b
✅ LM Studio for inference
✅ Works with any OpenAI-compatible endpoint

### 📌 Use Cases

✅ Research on graph-based reasoning
✅ Comparative retrieval models
✅ NLP and QA prototyping
✅ Lightweight RAG system demos
🤝 Contributing

We welcome PRs, issues, and suggestions!

Please:

Follow the folder structure
Include docstrings and inline comments
Add test cases for new components (if applicable)

### 🙌 Credits

Inspired by Microsoft Research’s GraphRAG methodology for global query-focused summarization:
📄 From Local to Global: A Graph RAG Approach to Query-Focused Summarization
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson
arXiv:2404.16130
DOI: 10.48550/arXiv.2404.16130

Uses DeepSeek’s distill model (deepseek-r1-distill-qwen-7b) for local LLM inference:
🔗 Hugging Face model card:
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Powered locally using LM Studio – open-source desktop LLM inference engine:
🌐 Official website:
https://lmstudio.ai

Built for experimental research and educational use.

 ### 📫 Contact

puranja@gmail.com for questions or collaboration.

