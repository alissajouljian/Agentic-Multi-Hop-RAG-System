"""
Query Planner: decomposes complex multi-hop questions into atomic
sub-questions and rewrites queries when retrieval confidence is low.
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI

from src.config import cfg
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)

DECOMPOSE_SYSTEM = """You are an expert question decomposer for a multi-hop \
question-answering system. Your job is to break a complex question into \
simpler, self-contained sub-questions that can each be answered by \
searching a Wikipedia corpus.

Rules:
1. Each sub-question must be independently searchable.
2. Order sub-questions so earlier answers inform later ones.
3. Use 1-{max_sub} sub-questions. Simple questions may need only 1.
4. Return ONLY valid JSON — no markdown fences, no commentary.

Output format (JSON array of strings):
["sub-question 1", "sub-question 2", ...]"""

REWRITE_SYSTEM = """You are a search-query rewriter. Given an original \
question, the context gathered so far, and a sub-question that failed to \
retrieve useful results, rewrite the sub-question to be more specific and \
searchable.

Return ONLY the rewritten question as a plain string — no JSON, no quotes, \
no commentary."""


class QueryPlanner:
    """LLM-powered query decomposition and rewriting."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.client = client or OpenAI(api_key=cfg.openai_api_key)
        self.counter = token_counter or TokenCounter()

    def decompose(self, question: str) -> list[str]:
        """
        Break a complex question into atomic sub-questions.

        Falls back to the original question if decomposition fails.
        """
        system = DECOMPOSE_SYSTEM.replace("{max_sub}", str(cfg.max_sub_questions))

        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            self.counter.update({
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })

            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if model adds them anyway
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            sub_questions = json.loads(raw)
            if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                log.info(
                    "Decomposed into %d sub-questions: %s",
                    len(sub_questions),
                    sub_questions,
                )
                return sub_questions[:cfg.max_sub_questions]

        except Exception as e:
            log.warning("Decomposition failed (%s), using original question", e)

        return [question]

    def rewrite(
        self,
        original_question: str,
        sub_question: str,
        context_so_far: str,
    ) -> str:
        """
        Rewrite a sub-question that yielded poor retrieval results.
        """
        user_prompt = (
            f"Original question: {original_question}\n\n"
            f"Context gathered so far:\n{context_so_far}\n\n"
            f"Sub-question to rewrite: {sub_question}\n\n"
            f"Rewritten sub-question:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            self.counter.update({
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })

            rewritten = resp.choices[0].message.content.strip().strip('"')
            log.info("Rewrote '%s' → '%s'", sub_question[:60], rewritten[:60])
            return rewritten

        except Exception as e:
            log.warning("Rewrite failed (%s), keeping original", e)
            return sub_question
