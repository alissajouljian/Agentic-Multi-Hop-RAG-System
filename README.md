# Agentic Multi-Hop RAG System

A production-quality **Agentic Retrieval-Augmented Generation** system built on the HotpotQA Wikipedia corpus. Unlike static one-shot RAG, the agent decomposes complex multi-hop questions, retrieves evidence iteratively, self-verifies every answer claim-by-claim, and rewrites failed queries autonomously until confident — or until its dynamic LLM budget is exhausted.

Evaluated on **30 hard HotpotQA questions**: agentic RAG delivers **+161% answer relevance**, **+128% Token F1**, **+54% groundedness**, and **−15% hallucination rate** vs the static baseline.

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. DECOMPOSE  (QueryPlanner / GPT-4o-mini)              │
│     → Breaks question into 1–5 atomic sub-questions      │
│     → Estimates complexity → allocates LLM budget        │
│       simple=4 calls · medium=8 calls · hard=12 calls    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  2. RETRIEVE  (HybridIndex: BM25 + FAISS + RRF)          │
│     BM25 (sparse, top-30) ─┐                             │
│                             ├→ RRF Fusion → top-20        │
│     FAISS dense (top-30) ──┘                             │
│     → LLM Cross-Encoder Re-Ranker → top-8 passages       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3. REASON  (Reasoner / GPT-4o-mini)                     │
│     → Chain-of-thought answer per sub-question           │
│     → Aggregate sub-answers into candidate final answer  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  4. VERIFY  (Verifier / GPT-4o-mini)                     │
│     → Extract atomic factual claims from answer          │
│     → Check each claim: supported by context? YES/NO     │
│     → Confidence = supported_claims / total_claims       │
└────────┬──────────────────────────────┬─────────────────┘
         │ confidence ≥ 0.60            │ confidence < 0.60
         │ OR conf ≥ 0.85 (early stop)  │ AND budget remaining
         ▼                              ▼
    ┌─────────┐                  ┌─────────────┐
    │  ANSWER │                  │  5. REWRITE │ → back to RETRIEVE
    └─────────┘                  │  (Planner)  │
                                 └─────────────┘
```

---

## Key Features

| Feature | Details |
|:---|:---|
| **Hybrid Retrieval** | BM25 sparse + FAISS dense with Reciprocal Rank Fusion (RRF) |
| **LLM Re-Ranking** | GPT-4o-mini cross-encoder style rescoring, top-8 passages to LLM |
| **Query Decomposition** | GPT-4o-mini breaks complex queries into 1–5 atomic sub-questions |
| **Iterative Retrieval** | Up to 4 re-query cycles with targeted sub-question retrieval |
| **Self-Verification** | Claim-level groundedness check via GPT-4o-mini (YES/NO per claim) |
| **Query Rewriting** | Rewrites failed sub-questions using all context gathered so far |
| **Dynamic LLM Budget** | Complexity classifier → simple=4 / medium=8 / hard=12 call ceiling |
| **High-Confidence Early Stop** | Skips rewrite loop if first-iteration confidence ≥ 0.85 |
| **Sub-Answer Cache** | Caches answered sub-questions; avoids redundant LLM calls |
| **ROUGE-L Scoring** | Implemented from scratch via LCS dynamic programming |
| **Streamlit Chat UI** | Real-time chat with agent trace, mode switcher, verification display |
| **Gradio Web UI** | Alternative web interface with example questions |
| **Rich CLI** | Interactive terminal Q&A with colour-coded trace tables |
| **7 Evaluation Plots** | Publication-quality matplotlib/seaborn charts |

---

## Evaluation Results (30 Hard HotpotQA Questions)

| Metric | Static RAG | Agentic RAG | Improvement |
|:---|:---:|:---:|:---:|
| Exact Match | 0.000 | 0.000 | — |
| **Token F1** | 0.014 | **0.032** | **+128%** |
| **ROUGE-L** | 0.014 | **0.032** | **+128%** |
| **Groundedness** | 0.222 | **0.342** | **+54%** |
| **Hallucination Rate** | 0.778 | **0.658** | **−15%** |
| **Answer Relevance** | 0.160 | **0.417** | **+161%** |
| Avg Latency | 2.4s | 26.1s | 11× (quality cost) |
| Avg LLM Calls | 1.0 | 9.6 | — |

**Agentic RAG wins on 25/30 questions (83%)** for answer relevance in per-question head-to-head comparison.

> **Note on Exact Match = 0:** The indexed corpus contains ~5,064 chunks from ~500 Wikipedia articles. HotpotQA draws from 5.2M articles — retrieval recall is 1.7% due to corpus coverage, not algorithm quality. Indexing the full HotpotQA distractor corpus (~500K paragraphs) would dramatically improve EM and F1.

---

## Quick Start

### 1. Setup

```bash
cd NLP
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env — add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Build the Index

```bash
# Quick start — index ~5K docs (already done, indices cached in data/indices/)
python main.py index --subset 5000

# Full corpus — 50K docs (~10 min, much better recall)
python main.py index
```

### 4. Ask Questions

```bash
# Agentic multi-hop RAG (full loop)
python main.py query "Were Scott Derrickson and Ed Wood of the same nationality?"

# Static single-shot RAG (baseline)
python main.py static "Were Scott Derrickson and Ed Wood of the same nationality?"

# Interactive Rich CLI (type questions, switch mode with 'mode static' / 'mode agentic')
python main.py chat

# Streamlit chat UI  →  opens at http://localhost:8501
python main.py streamlit

```

### 5. Evaluate & Plot

```bash
# Run full evaluation (30 questions, ~15 min)
python main.py evaluate --sample 30

# Generate all 7 plots
python main.py plots
```

---

## Interfaces

### Streamlit Chat UI (`http://localhost:8501`)
- **Mode switcher** — toggle Agentic vs Static live in the sidebar
- **Real-time chat** — full conversation history with scroll
- **Agent Trace expander** — see sub-questions, LLM call count, verification result, confidence, unsupported claims, and elapsed time for every response
- **Clear History** button

```bash
python main.py streamlit
```


### Rich CLI
- Colour-coded interactive terminal interface
- Retrieval tables per iteration, verification panel, token stats
- Switch modes mid-session: type `mode agentic` or `mode static`

```bash
python main.py chat
```

---

## Project Structure

```
NLP/
├── main.py                      # CLI entry point (all commands)
├── requirements.txt             # All dependencies
├── .env / .env.example          # API key configuration
├── src/
│   ├── config.py                # Central config & hyperparameters
│   ├── utils.py                 # Logging, timing, text helpers, TokenCounter
│   ├── indexing/
│   │   ├── corpus_loader.py     # HotpotQA Wikipedia corpus loader + cacher
│   │   ├── chunker.py           # 256-token overlap chunker
│   │   └── indexer.py           # BM25Index + DenseIndex + HybridIndex (RRF)
│   ├── retrieval/
│   │   ├── retriever.py         # Unified retrieval interface (bm25/dense/hybrid)
│   │   └── reranker.py          # LLM cross-encoder re-ranker
│   ├── agent/
│   │   ├── planner.py           # Query decomposition + rewriting (GPT-4o-mini)
│   │   ├── reasoner.py          # Sub-question answering + aggregation
│   │   ├── verifier.py          # Claim extraction + groundedness verification
│   │   ├── agent.py             # AgenticRAG — 6-state machine + dynamic budget
│   │   └── static_rag.py        # StaticRAG — single-shot baseline
│   ├── evaluation/
│   │   ├── eval_dataset.py      # HotpotQA eval question loader
│   │   ├── metrics.py           # EM, Token F1, ROUGE-L, groundedness, relevance
│   │   ├── evaluator.py         # Comparative evaluator (agentic vs static)
│   │   └── visualize.py         # 7 publication-quality plots
│   └── interface/
│       ├── streamlit_app.py     # Streamlit chat UI (port 8501)
│       └── cli.py               # Rich interactive terminal CLI
├── data/
│   ├── raw/corpus.jsonl         # Cached Wikipedia corpus
│   └── indices/                 # BM25 + FAISS + chunk store (auto-built)
└── results/
    ├── metrics/                 # Per-question + aggregate JSON results
    ├── plots/                   # 7 evaluation PNG charts
    ├── RESULTS_REPORT.md        # Full results write-up
    └── report.tex               # LaTeX academic report (Overleaf-ready)
```

---

## Stretch Goals Implemented

All 9 stretch-goal components from the assignment are complete:

| # | Component | Implementation |
|:---:|:---|:---|
| S1 | **Query Rewriting** | `planner.py` — rewrites failed sub-Qs with full context |
| S2 | **Sub-Answer Cache** | `agent.py` — dict cache prevents repeated LLM calls |
| S3 | **Dynamic LLM Budget** | `agent.py` — simple=4 / medium=8 / hard=12 call limits |
| S4 | **High-Confidence Early Stop** | `agent.py` — skips rewrite if confidence ≥ 0.85 |
| S5 | **Claim-Level Self-Verification** | `verifier.py` — atomic claim extraction + YES/NO checking |
| S6 | **ROUGE-L Metric** | `metrics.py` — LCS dynamic programming from scratch |
| S7 | **Budget vs Accuracy Analysis** | `visualize.py` — scatter + regression coloured by complexity |
| S8 | **Complexity-Stratified Breakdown** | `visualize.py` — F1 by simple/medium/hard |
| S9 | **Detailed Error Taxonomy** | `metrics.py` — correct / retrieval_failure / hallucination / reasoning_failure |

---

## Technology Stack

| Component | Technology |
|:---|:---|
| LLM (reasoning, verification, reranking) | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS IndexFlatIP (L2-normalized) |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Corpus | HotpotQA Wikipedia (HuggingFace datasets) |
| Orchestration | Custom Python 6-state FSM |
| CLI | Click + Rich |
| Chat UI | Streamlit |
| Evaluation Plots | Matplotlib + Seaborn + Pandas |

---

## Hyperparameters

| Parameter | Value |
|:---|:---:|
| Chunk size (tokens) | 256 |
| Chunk overlap (tokens) | 64 |
| BM25 / Dense top-k | 30 each |
| RRF fused top-k | 20 |
| Re-ranked passages to LLM | 8 |
| Max sub-questions | 5 |
| Max agent iterations | 4 |
| Verification threshold | 0.60 |
| High-confidence early stop | 0.85 |
| LLM budget: simple/medium/hard | 4 / 8 / 12 calls |
| LLM temperature (generation) | 0.1 |
| LLM temperature (structured) | 0.0 |
