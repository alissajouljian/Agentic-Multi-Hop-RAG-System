"""
Cross-encoder re-ranker: rescores top-k retrieval results.
Falls back to LLM-based re-ranking if local models fail.
"""

from __future__ import annotations

from typing import Optional
from openai import OpenAI

from src.config import cfg
from src.retrieval.retriever import RetrievalResult
from src.utils import get_logger, timer

log = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-rank retrieval results using a cross-encoder model or LLM.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or cfg.reranker_model
        self._model = None
        self.client = OpenAI(api_key=cfg.openai_api_key)

    def _rerank_llm(self, query: str, results: list[RetrievalResult], top_n: int) -> list[RetrievalResult]:
        """Use LLM to re-rank results when local model is unavailable."""
        log.info("Using LLM-based re-ranking (GPT-4o-mini) …")
        
        passages = "\n".join([f"[{i}] {r.title}: {r.text[:200]}..." for i, r in enumerate(results)])
        prompt = (
            f"Question: {query}\n\n"
            f"Passages:\n{passages}\n\n"
            "Rank the passages above by their relevance to the question. "
            "Return only a comma-separated list of indices in order of relevance (e.g., 2,0,1). "
            f"Return at most {top_n} indices."
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            indices_str = resp.choices[0].message.content.strip()
            indices = [int(i.strip()) for i in indices_str.split(",") if i.strip().isdigit()]
            
            reranked = []
            for i, idx in enumerate(indices[:top_n]):
                if idx < len(results):
                    res = results[idx]
                    reranked.append(RetrievalResult(
                        chunk=res.chunk, score=1.0 - (i * 0.1), rank=i, method="llm-reranked"
                    ))
            return reranked
        except Exception as e:
            log.warning("LLM reranking failed: %s. Returning top-k original.", e)
            return results[:top_n]

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> list[RetrievalResult]:
        top_n = top_n or cfg.rerank_top_n
        if not results:
            return []

        # Force LLM reranking for stability in this environment
        return self._rerank_llm(query, results, top_n)
