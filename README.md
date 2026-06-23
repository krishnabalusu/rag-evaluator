# RAG Evaluator

> A RAG pipeline built from scratch with a proper evaluation framework using RAGAS — because "it looks good" is not a good enough reason to ship.

---

## What this project is about

I kept seeing the same pattern in RAG tutorials: build the pipeline, ask it a few questions, answers look okay, done. Nobody actually measures whether the system is working well or just *seems* to be working.

That's the problem this project solves. I built a full RAG pipeline — every stage written explicitly, not hidden inside a framework — and then built an evaluation layer on top using RAGAS to actually measure performance with real metrics. Then I ran two different configurations against each other to see which one wins and why.

The result is a system where you can change a parameter (like chunk size), run the eval, and get actual numbers back instead of just vibes.

---

## What the evaluation found

I tested two chunk sizes on 10 questions generated from 21 Wikipedia AI/ML articles:

```
==============================================================
        Benchmark — chunk=512 vs chunk=256
==============================================================
  Metric              chunk=512    chunk=256    Difference
  Faithfulness          0.9286       0.5867       +0.34
  Answer Relevancy      0.6554       0.4100       +0.25
  Context Recall        0.8000       0.5556       +0.24
  Overall               0.7947       0.5174       +0.28
==============================================================
```

Larger chunks (512) beat smaller chunks (256) on every metric by a big margin. The reason makes sense once you think about it — when you split a Wikipedia article into 256-character pieces, you end up cutting sentences in half. The retriever finds the right chunk but the chunk doesn't have enough context to form a good answer. Larger chunks keep more context together.

---

## The 3 metrics I used

### 🔒 Faithfulness
This checks whether the answer the LLM gave is actually backed up by what was retrieved. If the model says something that isn't in the retrieved chunks, faithfulness goes down. High faithfulness means the model is not hallucinating.

> *"Did the model only use what it was given?"*

### 🎯 Answer Relevancy
The answer can be perfectly faithful — only using retrieved text — but still not actually answer the question. This metric checks whether the response addresses what was asked.

> *"Did the model answer the right question?"*

### 🔍 Context Recall
This checks whether the retriever actually found the chunks needed to answer the question. If key information existed in the documents but the retriever missed it, this score drops.

> *"Did the retriever find what it needed?"*

---

## How the pipeline works

Instead of wrapping everything in LangChain and calling it done, I built each stage separately so you can see exactly what's happening at every step.

```
Your Documents (PDF / DOCX / TXT / HTML / Markdown)
          │
          ▼
   [ ingestion/document_loader.py ]
   Reads each file format separately, keeps source metadata
          │
          ▼
   [ preprocessing/text_cleaner.py ]
   Fixes encoding, removes noise, normalizes whitespace
          │
          ▼
   [ chunking/chunker.py ]
   Splits text into overlapping chunks, metadata preserved
          │
          ▼
   [ embeddings/embedder.py ]
   Converts each chunk to a 384-dim vector using all-MiniLM-L6-v2
          │
          ▼
   [ vectorstore/vector_db.py ]
   Stores vectors in ChromaDB with cosine similarity
          │
          ▼
   [ retrieval/retriever.py ]
   Embeds the query, finds similar chunks, filters low scores
          │
          ▼
   [ generation/generator.py ]
   Sends retrieved chunks + question to LLM, gets answer
          │
          ▼
   [ evaluation/evaluator.py ]
   Scores with RAGAS, saves results, generates comparison
```

Each file is one thing. Nothing is hidden.

---

## Tech stack

| What | How |
|---|---|
| Pipeline | Pure Python — no LangChain in the core |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free) |
| Vector store | ChromaDB (local, persistent) |
| LLM | Groq `llama-3.3-70b-versatile` (free API) |
| Evaluation | RAGAS 0.4.x |
| Document formats | PDF, DOCX, TXT, HTML, Markdown |

I used Groq because it's free and fast. The embedding model runs fully locally so there's no API call needed for indexing.

---

## Project structure

```
rag-evaluator/
├── config/
│   └── settings.py              # All tunable parameters in one place
├── ingestion/
│   └── document_loader.py       # Separate loader for each file format
├── preprocessing/
│   └── text_cleaner.py          # Text normalization before chunking
├── chunking/
│   └── chunker.py               # Splits docs into overlapping chunks
├── embeddings/
│   └── embedder.py              # Wraps sentence-transformers
├── vectorstore/
│   └── vector_db.py             # ChromaDB add/search/clear
├── retrieval/
│   └── retriever.py             # Query → embed → search → filter
├── generation/
│   └── generator.py             # LLM call with anti-hallucination prompt
├── pipeline/
│   └── rag_pipeline.py          # Connects all stages: index() + query()
├── evaluation/
│   ├── langchain_pipeline.py    # LangChain adapter for RAGAS
│   ├── testset_generator.py     # Generates QA pairs from actual documents
│   ├── evaluator.py             # Runs RAGAS scoring
│   └── report.py                # Comparison report + resume bullet
├── results/                     # Scores saved here as JSON
├── demo.py                      # Quick demo of the full pipeline
├── download_wiki.py             # Downloads 21 Wikipedia AI/ML articles
├── index_wiki.py                # Indexes them into ChromaDB
└── run_evaluation.py            # Main entry point for evaluation
```

---

## Setup

**1. Install dependencies**
```bash
git clone https://github.com/krishnabalusu/rag-evaluator.git
cd rag-evaluator
pip install -r requirements.txt
```

**2. Get a free Groq API key**

Go to [console.groq.com](https://console.groq.com), sign up, create an API key. It's free.

**3. Set up your .env file**
```bash
cp .env.example .env
```

Then open `.env` and fill in your key:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

---

## Running it

### Quick demo — see the RAG pipeline answer questions
```bash
python demo.py
```

### Full evaluation — download docs, index, run benchmarks
```bash
# Download the Wikipedia articles used as knowledge base
python download_wiki.py

# Index them into ChromaDB
python index_wiki.py

# Run evaluation on both configs and compare
python run_evaluation.py
```

### Options if you've already run it once
```bash
python run_evaluation.py --testset-only   # Just regenerate QA pairs
python run_evaluation.py --eval-only      # Skip indexing, reuse testset
python run_evaluation.py --compare        # Just print the comparison
```

---

## Knowledge base

21 Wikipedia articles on AI and ML topics:

Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, Large Language Model, Transformer (machine learning), Retrieval-Augmented Generation, Vector Database, Word Embedding, Reinforcement Learning, Neural Network, Convolutional Neural Network, Recurrent Neural Network, Attention Mechanism, Transfer Learning, BERT, GPT, Semantic Search, Information Retrieval, Data Mining, Text Mining

---

## What's next

Things I want to add:
- Reranking with a cross-encoder on top of the retriever
- Hybrid search (keyword + semantic combined)
- FastAPI layer to turn this into a REST API
- More evaluation questions (currently 10, want 50+)

---

## License

MIT
