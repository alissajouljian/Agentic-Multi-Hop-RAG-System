"""
StaticRAG: single-shot retrieve-and-generate baseline.

No query decomposition, no re-query, no self-verification.
"""

from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from src.config import cfg
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.agent.agent import AgentTrace
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)

STATIC_SYSTEM = """You are a helpful question-answering assistant. Given a \
question and a set of retrieved passages, provide a concise, accurate answer.

Rules:
1. Answer ONLY from the provided passages.
2. If no passage contains the answer, say "I don't have enough information \
to answer this question."
3. Cite sources using [Article Title] notation.
4. Be concise — aim for 1-3 sentences."""


class StaticRAG:
    """
    Single-shot RAG baseline.

    Pipeline: query → hybrid retrieve → re-rank → generate answer
    No decomposition, no verification loop, no re-query.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.token_counter = TokenCounter()
        self.client   = client or OpenAI(api_key=cfg.openai_api_key)
        self.retriever = retriever
        self.reranker  = reranker or CrossEncoderReranker()

    def query(self, question: str) -> tuple[str, AgentTrace]:
        """
        Single-shot RAG: retrieve → re-rank → answer.

        Returns:
            Tuple of (answer, trace).
        """
        self.token_counter.prompt_tokens = 0
        self.token_counter.completion_tokens = 0
        self.token_counter.calls = 0

        t0    = time.perf_counter()
        trace = AgentTrace(original_question=question, sub_questions=[question])

        results  = self.retriever.search(question, top_k=cfg.hybrid_top_k)
        trace.total_retrievals = 1

        reranked = self.reranker.rerank(question, results, top_n=cfg.rerank_top_n)
        trace.passages = {0: reranked}

        passage_text_list = [
            f"[{r.title}]: {r.text}" for r in reranked
        ]

        trace.iterations.append({
            "iteration":    0,
            "sub_results": [{
                "sub_question":  question,
                "num_retrieved": len(results),
                "num_reranked":  len(reranked),
                "top_titles":    [r.title for r in reranked],
                "passage_texts": passage_text_list,
            }],
        })

        passage_text = "\n\n".join(
            f"[Passage {i+1} — {r.title}]\n{r.text}"
            for i, r in enumerate(reranked)
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Passages:\n{passage_text}\n\n"
            f"Answer:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": STATIC_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            self.token_counter.update({
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })
            answer = resp.choices[0].message.content.strip()

        except Exception as e:
            log.error("Static RAG generation failed: %s", e)
            answer = "Unable to generate an answer."

        trace.final_answer    = answer
        trace.total_llm_calls = self.token_counter.calls
        trace.elapsed_seconds = time.perf_counter() - t0
        trace.token_summary   = self.token_counter.summary()

        log.info(
            "Static RAG finished in %.1fs | %s",
            trace.elapsed_seconds,
            trace.token_summary,
        )

        return answer, trace
