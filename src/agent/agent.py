"""
AgenticRAG: orchestrates decomposition, iterative retrieval,
reasoning, and self-verification.

State machine: DECOMPOSE → RETRIEVE → REASON → VERIFY → (REWRITE →) ANSWER
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from openai import OpenAI

from src.config import cfg
from src.agent.planner import QueryPlanner
from src.agent.reasoner import Reasoner
from src.agent.verifier import Verifier, VerificationResult
from src.retrieval.retriever import Retriever, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)


class AgentState(Enum):
    DECOMPOSE = auto()
    RETRIEVE = auto()
    REASON = auto()
    VERIFY = auto()
    REWRITE = auto()
    ANSWER = auto()
    FAILED = auto()


_COMPLEXITY_BUDGET = {
    "simple": 4,
    "medium": 8,
    "hard":   12,
}

_HIGH_CONFIDENCE_THRESHOLD = 0.85   # skip re-query if already very confident


@dataclass
class AgentTrace:
    """Complete trace of agent execution for observability."""

    original_question: str = ""
    sub_questions: list[str] = field(default_factory=list)
    iterations: list[dict] = field(default_factory=list)
    final_answer: str = ""
    verification: Optional[dict] = None
    total_retrievals: int = 0
    total_llm_calls: int = 0
    elapsed_seconds: float = 0.0
    token_summary: str = ""
    complexity: str = "medium"
    dynamic_budget: int = cfg.max_llm_calls
    passages: dict[int, list[RetrievalResult]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "original_question":  self.original_question,
            "sub_questions":      self.sub_questions,
            "iterations":         self.iterations,
            "final_answer":       self.final_answer,
            "verification":       self.verification,
            "total_retrievals":   self.total_retrievals,
            "total_llm_calls":    self.total_llm_calls,
            "elapsed_seconds":    self.elapsed_seconds,
            "token_summary":      self.token_summary,
            "complexity":         self.complexity,
            "dynamic_budget":     self.dynamic_budget,
        }
        return d


def _classify_complexity(question: str, num_sub_questions: int) -> str:
    """
    Estimate question complexity to set a dynamic LLM call budget.

    Heuristic rules (no extra LLM call needed):
      - 1 sub-question or question < 10 words  → simple
      - 3+ sub-questions or keywords like
        "compared", "differ", "relationship"   → hard
      - otherwise                              → medium
    """
    q_lower = question.lower()
    hard_kws = {"compared", "differ", "relationship", "both", "between",
                "what did", "how many", "which of", "in common"}
    if num_sub_questions >= 3 or any(kw in q_lower for kw in hard_kws):
        return "hard"
    if num_sub_questions == 1 or len(question.split()) < 10:
        return "simple"
    return "medium"


class AgenticRAG:
    """Multi-hop agentic RAG with iterative retrieval and self-verification."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.token_counter = TokenCounter()
        self.client = client or OpenAI(api_key=cfg.openai_api_key)

        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.planner = QueryPlanner(client=self.client, token_counter=self.token_counter)
        self.reasoner = Reasoner(client=self.client, token_counter=self.token_counter)
        self.verifier = Verifier(client=self.client, token_counter=self.token_counter)

        self._answer_cache: dict[str, str] = {}

    def query(
        self,
        question: str,
        max_iterations: Optional[int] = None,
    ) -> tuple[str, AgentTrace]:
        """
        Run the full agentic RAG pipeline.

        Args:
            question: The user's complex multi-hop question.
            max_iterations: Max re-query cycles (default from config).

        Returns:
            Tuple of (final_answer, trace).
        """
        self._answer_cache = {}
        self.token_counter.prompt_tokens = 0
        self.token_counter.completion_tokens = 0
        self.token_counter.calls = 0

        t0 = time.perf_counter()
        trace = AgentTrace(original_question=question)
        state = AgentState.DECOMPOSE

        sub_questions: list[str] = []
        sub_answers: dict[int, str] = {}
        all_passages: dict[int, list[RetrievalResult]] = {}
        candidate_answer = ""
        iteration = 0

        while state not in (AgentState.ANSWER, AgentState.FAILED):

            if state == AgentState.DECOMPOSE:
                log.info("── DECOMPOSE (iteration %d) ──", iteration)
                sub_questions = self.planner.decompose(question)
                trace.sub_questions = sub_questions

                complexity = _classify_complexity(question, len(sub_questions))
                trace.complexity = complexity
                dynamic_budget = _COMPLEXITY_BUDGET[complexity]
                trace.dynamic_budget = dynamic_budget
                log.info("Complexity: %s → budget %d LLM calls", complexity, dynamic_budget)

                state = AgentState.RETRIEVE

            if self.token_counter.calls >= trace.dynamic_budget:
                log.warning("Dynamic LLM budget exhausted (%d calls)", self.token_counter.calls)
                state = AgentState.ANSWER
                if not candidate_answer and sub_answers:
                    pairs = [(sub_questions[i], sub_answers[i]) for i in sorted(sub_answers)]
                    candidate_answer = self.reasoner.aggregate(question, pairs)
                elif not candidate_answer:
                    candidate_answer = "Budget exhausted before generating an answer."
                break

            if iteration >= (max_iterations or cfg.max_agent_iterations):
                log.warning("Max iterations reached (%d)", iteration)
                state = AgentState.ANSWER
                if not candidate_answer and sub_answers:
                    pairs = [(sub_questions[i], sub_answers[i]) for i in sorted(sub_answers)]
                    candidate_answer = self.reasoner.aggregate(question, pairs)
                break

            if state == AgentState.RETRIEVE:
                log.info("── RETRIEVE (iteration %d) ──", iteration)
                iter_data = {"iteration": iteration, "sub_results": []}

                for i, sq in enumerate(sub_questions):
                    if i in sub_answers and sub_answers[i] != "INSUFFICIENT_CONTEXT":
                        continue

                    results = self.retriever.search(sq, top_k=cfg.hybrid_top_k)
                    trace.total_retrievals += 1

                    reranked = self.reranker.rerank(sq, results, top_n=cfg.rerank_top_n)
                    all_passages[i] = reranked

                    iter_data["sub_results"].append({
                        "sub_question":  sq,
                        "num_retrieved": len(results),
                        "num_reranked":  len(reranked),
                        "top_titles":    [r.title for r in reranked],
                    })

                trace.iterations.append(iter_data)
                trace.passages = dict(all_passages)
                state = AgentState.REASON

            elif state == AgentState.REASON:
                log.info("── REASON (iteration %d) ──", iteration)

                for i, sq in enumerate(sub_questions):
                    if i in sub_answers and sub_answers[i] != "INSUFFICIENT_CONTEXT":
                        continue

                    cache_key = sq.lower().strip()
                    if cache_key in self._answer_cache:
                        log.info("Cache hit for sub-question: %s", sq[:60])
                        sub_answers[i] = self._answer_cache[cache_key]
                        continue

                    passages = all_passages.get(i, [])
                    if not passages:
                        sub_answers[i] = "INSUFFICIENT_CONTEXT"
                        continue

                    answer = self.reasoner.answer_sub_question(sq, passages)
                    sub_answers[i] = answer
                    if answer != "INSUFFICIENT_CONTEXT":
                        self._answer_cache[cache_key] = answer

                pairs = [(sub_questions[i], sub_answers[i]) for i in sorted(sub_answers)]
                candidate_answer = self.reasoner.aggregate(question, pairs)
                state = AgentState.VERIFY

            elif state == AgentState.VERIFY:
                log.info("── VERIFY (iteration %d) ──", iteration)

                all_context = "\n\n".join(
                    f"[{r.title}]: {r.text}"
                    for passages in all_passages.values()
                    for r in passages
                )

                vr: VerificationResult = self.verifier.verify(
                    candidate_answer, all_context
                )
                trace.verification = {
                    "confidence":    vr.confidence,
                    "total_claims":  vr.total_claims,
                    "supported":     vr.supported_claims,
                    "unsupported":   vr.unsupported,
                    "passed":        vr.passed,
                }

                if vr.passed:
                    log.info("✓ Verification PASSED (confidence=%.2f)", vr.confidence)
                    state = AgentState.ANSWER
                elif vr.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
                    log.info(
                        "✓ Early stop: confidence %.2f ≥ %.2f (high-confidence threshold)",
                        vr.confidence, _HIGH_CONFIDENCE_THRESHOLD,
                    )
                    trace.verification["passed"] = True   # mark as passed
                    state = AgentState.ANSWER
                else:
                    log.info(
                        "✗ Verification FAILED (confidence=%.2f), rewriting …",
                        vr.confidence,
                    )
                    state = AgentState.REWRITE
                    iteration += 1

            elif state == AgentState.REWRITE:
                log.info("── REWRITE (iteration %d) ──", iteration)

                context_so_far = "\n".join(
                    f"Q: {sub_questions[i]}\nA: {sub_answers.get(i, 'N/A')}"
                    for i in range(len(sub_questions))
                )

                for i, sq in enumerate(sub_questions):
                    answer = sub_answers.get(i, "INSUFFICIENT_CONTEXT")
                    if answer == "INSUFFICIENT_CONTEXT":
                        sub_questions[i] = self.planner.rewrite(
                            question, sq, context_so_far
                        )
                        sub_answers.pop(i, None)

                state = AgentState.RETRIEVE

        trace.final_answer   = candidate_answer or "Unable to determine an answer."
        trace.total_llm_calls = self.token_counter.calls
        trace.elapsed_seconds = time.perf_counter() - t0
        trace.token_summary   = self.token_counter.summary()
        trace.passages        = all_passages   # final snapshot

        log.info(
            "Agent finished in %.1fs | complexity=%s | budget=%d | %s",
            trace.elapsed_seconds,
            trace.complexity,
            trace.dynamic_budget,
            trace.token_summary,
        )

        return trace.final_answer, trace
