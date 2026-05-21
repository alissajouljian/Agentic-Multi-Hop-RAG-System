"""
Reasoner: synthesises answers from retrieved passages using chain-of-thought
prompting with citation instructions.
"""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from src.config import cfg
from src.retrieval.retriever import RetrievalResult
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)

SUB_ANSWER_SYSTEM = """You are a precise research assistant. Given a question \
and a set of retrieved passages, provide a concise factual answer.

Rules:
1. Answer ONLY from the provided passages. If no passage contains the answer, \
say "INSUFFICIENT_CONTEXT".
2. Be concise — one or two sentences.
3. Cite the passage title in brackets, e.g. [Article Title].
4. Think step-by-step before answering."""

AGGREGATE_SYSTEM = """You are an expert synthesiser. Given a complex question \
and a set of intermediate answers to its sub-questions, produce a final, \
comprehensive answer.

Rules:
1. Combine the intermediate answers coherently.
2. Resolve any contradictions by preferring answers with citations.
3. Cite sources using [Article Title] notation.
4. Be concise but complete — aim for 2-4 sentences.
5. If intermediate answers are insufficient, say so explicitly."""


class Reasoner:
    """LLM-powered synthesis of sub-answers and final aggregation."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.client = client or OpenAI(api_key=cfg.openai_api_key)
        self.counter = token_counter or TokenCounter()

    def answer_sub_question(
        self,
        sub_question: str,
        passages: list[RetrievalResult],
    ) -> str:
        """
        Generate an answer for a single sub-question given retrieved passages.
        """
        passage_text = "\n\n".join(
            f"[Passage {i+1} — {p.title}]\n{p.text}"
            for i, p in enumerate(passages)
        )

        user_prompt = (
            f"Question: {sub_question}\n\n"
            f"Retrieved Passages:\n{passage_text}\n\n"
            f"Answer:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": SUB_ANSWER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            self.counter.update({
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })

            answer = resp.choices[0].message.content.strip()
            log.debug("Sub-answer for '%s': %s", sub_question[:50], answer[:100])
            return answer

        except Exception as e:
            log.error("Sub-question answering failed: %s", e)
            return "INSUFFICIENT_CONTEXT"

    def aggregate(
        self,
        original_question: str,
        sub_qa_pairs: list[tuple[str, str]],
    ) -> str:
        """
        Aggregate intermediate sub-question answers into a final answer.

        Args:
            original_question: The user's original complex question.
            sub_qa_pairs: List of (sub_question, sub_answer) tuples.

        Returns:
            Final synthesised answer string.
        """
        intermediate = "\n\n".join(
            f"Sub-question {i+1}: {q}\nIntermediate answer: {a}"
            for i, (q, a) in enumerate(sub_qa_pairs)
        )

        user_prompt = (
            f"Original question: {original_question}\n\n"
            f"Intermediate results:\n{intermediate}\n\n"
            f"Final answer:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": AGGREGATE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            self.counter.update({
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })

            answer = resp.choices[0].message.content.strip()
            log.info("Aggregated answer: %s", answer[:120])
            return answer

        except Exception as e:
            log.error("Aggregation failed: %s", e)
            for _, a in sub_qa_pairs:
                if a != "INSUFFICIENT_CONTEXT":
                    return a
            return "Unable to generate an answer from the retrieved information."
