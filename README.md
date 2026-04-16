<<<<<<< HEAD
# Youtube-Rag-chatbot
=======
# 🎬 YouTube Video RAG Chatbot

A fully local, open-source Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions about any YouTube video's content.

## ✨ Features

- **Transcript Ingestion** — Fetches and preprocesses YouTube captions automatically
- **Configurable Chunking** — Fixed-size (200/500/1000 tokens) and sentence-boundary strategies
- **Multiple Embedding Models** — MiniLM, MPNet, E5 for comparative evaluation
- **FAISS Vector Store** — Fast local similarity search with cosine similarity
- **Cross-Encoder Reranking** — Optional two-stage retrieval for higher precision
- **Local LLM Generation** — Mistral-7B and Llama-2-7B via Ollama (no API keys needed)
- **Conversational Memory** — Multi-turn dialogue with context-aware follow-ups
- **Timestamp Citations** — Every answer includes clickable `[MM:SS - MM:SS]` references
- **Streamlit Chat UI** — Premium dark-themed interface with glassmorphism design
- **Full Evaluation Suite** — Precision@K, Recall@K, MRR, BLEU, ROUGE, BERTScore, RAGAS Faithfulness
- **48-Configuration Ablation** — Systematic grid search across all hyperparameters

## 🏗️ Architecture

```
Video URL → Transcript Fetch → Preprocess → Chunk → Embed → FAISS Index
                                                                  ↓
User Question → Embed Query → FAISS Retrieve → (Rerank) → LLM Generate → Answer + Citations
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from [ollama.com](https://ollama.com) and pull models:
   ```bash
   ollama pull mistral
   ollama pull llama2:7b  # for ablation comparison
   ```

### Installation

```bash
cd yt-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Chat Interface

```bash
streamlit run app/streamlit_app.py
```

1. Paste a YouTube URL in the sidebar
2. Click **Ingest Video**
3. Start chatting!

## 📁 Project Structure

```
yt-rag-chatbot/
├── config.py                     # Central configuration
├── src/
│   ├── transcript/               # Fetch + preprocess YouTube captions
│   ├── chunking/                 # Fixed-size & sentence-boundary chunking
│   ├── embedding/                # Sentence-transformer embeddings
│   ├── vectorstore/              # FAISS index management
│   ├── retrieval/                # Retriever + cross-encoder reranker
│   ├── generation/               # LLM, prompts, conversational RAG chain
│   └── pipeline.py               # End-to-end orchestrator
├── app/                          # Streamlit chat interface
├── evaluation/                   # Eval framework + ablation runner
│   ├── retrieval_eval.py         # Precision@K, Recall@K, Hit Rate, MRR
│   ├── generation_eval.py        # BLEU, ROUGE, BERTScore
│   ├── faithfulness_eval.py      # RAGAS + fact-level precision
│   ├── ablation.py               # 48-config grid search
│   └── dataset/                  # QA pairs + annotation helper
├── tests/                        # Pytest test suite
└── data/                         # Transcripts + FAISS indices
```

## 🔬 Evaluation

### Generate QA Pairs

```bash
python evaluation/dataset/annotation_helper.py \
  --video-url "https://youtube.com/watch?v=VIDEO_ID" \
  --domain cs_lectures
```

### Run Retrieval Evaluation

```bash
python evaluation/retrieval_eval.py \
  --video-id VIDEO_ID \
  --embedding minilm \
  --chunk-strategy fixed \
  --chunk-size 500
```

### Run Full Ablation Study

```bash
python evaluation/ablation.py \
  --video-ids VIDEO_ID_1 VIDEO_ID_2 \
  --skip-ingestion \  # if indices already built
  --output-prefix my_ablation
```

For a quick test with 2 configurations:
```bash
python evaluation/ablation.py --video-ids VIDEO_ID --quick
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 Research Questions

| RQ | Question | Metrics |
|----|----------|---------|
| RQ1 | Pipeline accuracy across domains | Precision@K, BERTScore |
| RQ2 | Chunking strategy comparison | Hit Rate, MRR |
| RQ3 | Embedding model impact | Precision@K, Recall@K |
| RQ4 | Hallucination on fact-dense domains | RAGAS Faithfulness, Fact Precision |

## ⚙️ Configuration

All hyperparameters are controlled via `config.py`:

```python
from config import PipelineConfig, ChunkingConfig, RetrievalConfig, GenerationConfig

config = PipelineConfig(
    embedding_model="minilm",           # minilm | mpnet | e5
    chunking=ChunkingConfig(
        strategy="fixed",               # fixed | sentence
        chunk_size=500,                  # 200 | 500 | 1000
    ),
    retrieval=RetrievalConfig(
        use_reranker=True,               # FAISS-only vs FAISS+cross-encoder
        top_k=5,
    ),
    generation=GenerationConfig(
        model_name="mistral",            # mistral | llama2
        temperature=0.1,
    ),
)
```

## 📝 License

MIT
>>>>>>> 908aad4 (Initial commit)
