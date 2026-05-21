"""
Central configuration for the Agentic Multi-Hop RAG System.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Immutable-ish configuration for the entire system."""

    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    llm_model:       str = os.getenv("LLM_MODEL",       "gpt-4o-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model:  str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    nli_hf_model:    str = os.getenv(
        "NLI_HF_MODEL",
        "microsoft/deberta-v3-base-mnli-fever-anli",
    )

    chunk_size:    int = 256
    chunk_overlap: int = 64

    bm25_top_k:   int = 30
    dense_top_k:  int = 30
    hybrid_top_k: int = 20
    rerank_top_n: int = 8

    max_agent_iterations:   int   = 4
    max_sub_questions:      int   = 5
    verification_threshold: float = 0.60
    max_llm_calls:          int   = 12

    high_confidence_threshold: float = 0.85

    complexity_budget_simple: int = 4
    complexity_budget_medium: int = 8
    complexity_budget_hard:   int = 12

    corpus_subset_size: int = 50_000
    hotpotqa_split:     str = "train"

    eval_sample_size: int = 100
    eval_split:       str = "validation"

    data_dir:      Path = PROJECT_ROOT / "data"
    raw_dir:       Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir:     Path = PROJECT_ROOT / "data" / "indices"
    results_dir:   Path = PROJECT_ROOT / "results"
    metrics_dir:   Path = PROJECT_ROOT / "results" / "metrics"
    plots_dir:     Path = PROJECT_ROOT / "results" / "plots"

    temperature: float = 0.1
    max_tokens:  int   = 1024

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.raw_dir, self.processed_dir, self.index_dir,
            self.results_dir, self.metrics_dir, self.plots_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
