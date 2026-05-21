"""
Retriever wrappers: BM25, Dense, and Hybrid retrieval with a clean API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import cfg
from src.indexing.chunker import Chunk
from src.indexing.indexer import HybridIndex
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved passage with metadata."""

    chunk: Chunk
    score: float
    rank: int
    method: str  # "bm25", "dense", "hybrid", "reranked"

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def title(self) -> str:
        return self.chunk.title

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"title='{self.title[:40]}…', method={self.method})"
        )


class Retriever:
    """
    Unified retrieval interface wrapping the HybridIndex.

    Supports BM25-only, dense-only, or hybrid (RRF) modes.
    """

    def __init__(self, index: HybridIndex) -> None:
        self.index = index

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k passages for the given query.

        Args:
            query: The search query.
            top_k: Number of results to return.
            mode: "bm25", "dense", or "hybrid".

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        top_k = top_k or cfg.hybrid_top_k

        if mode == "bm25":
            raw = self.index.bm25.search(query, top_k=top_k)
            results = [
                RetrievalResult(
                    chunk=self.index.chunk_store.get_by_index(idx),
                    score=score,
                    rank=rank,
                    method="bm25",
                )
                for rank, (idx, score) in enumerate(raw)
            ]
        elif mode == "dense":
            raw = self.index.dense.search(query, top_k=top_k)
            results = [
                RetrievalResult(
                    chunk=self.index.chunk_store.get_by_index(idx),
                    score=score,
                    rank=rank,
                    method="dense",
                )
                for rank, (idx, score) in enumerate(raw)
            ]
        else:
            raw = self.index.search(query, top_k=top_k)
            results = [
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    method="hybrid",
                )
                for rank, (chunk, score) in enumerate(raw)
            ]

        log.debug("Retrieved %d results for query: '%.60s…'", len(results), query)
        return results
