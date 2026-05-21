"""
Evaluator: runs AgenticRAG and StaticRAG over the evaluation set,
computes all metrics, and saves comparison results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import cfg
from src.agent.agent import AgenticRAG, AgentTrace
from src.agent.static_rag import StaticRAG
from src.agent.verifier import Verifier
from src.evaluation.eval_dataset import load_eval_dataset
from src.evaluation.metrics import (
    QuestionMetrics,
    AggregateMetrics,
    exact_match,
    token_f1,
    rouge_l_score,
    answer_relevance_llm,
    classify_error,
    compute_aggregates,
)
from src.retrieval.retriever import Retriever, RetrievalResult
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)


class Evaluator:
    """
    Run comparative evaluation of AgenticRAG vs StaticRAG.

    Produces:
    - Per-question metrics (JSON)
    - Aggregate comparison table (JSON)
    - Raw traces for failure analysis
    """

    def __init__(
        self,
        retriever: Retriever,
        agentic: Optional[AgenticRAG] = None,
        static: Optional[StaticRAG] = None,
    ) -> None:
        self.retriever = retriever
        self.agentic = agentic or AgenticRAG(retriever=retriever)
        self.static = static or StaticRAG(retriever=retriever)
        self.verifier = Verifier()
        self.eval_counter = TokenCounter()

    def _build_passage_context(self, trace: AgentTrace) -> str:
        """Build passage-text context string from a trace for groundedness scoring."""
        passage_texts: list[str] = []

        if hasattr(trace, "passages") and trace.passages:
            for results in trace.passages.values():
                for r in results:
                    passage_texts.append(f"[{r.title}]: {r.text}")
            if passage_texts:
                return "\n\n".join(passage_texts)

        for it in trace.iterations:
            for sr in it.get("sub_results", []):
                for txt in sr.get("passage_texts", []):
                    passage_texts.append(txt)

        if passage_texts:
            return "\n\n".join(passage_texts)

        for it in trace.iterations:
            for sr in it.get("sub_results", []):
                sq = sr.get("sub_question", trace.original_question)
                try:
                    results = self.retriever.search(sq, top_k=cfg.rerank_top_n)
                    for r in results:
                        passage_texts.append(f"[{r.title}]: {r.text}")
                except Exception:
                    pass

        return "\n\n".join(passage_texts)

    def _build_retrieval_recall(
        self,
        trace: AgentTrace,
        gold_titles: list[str],
    ) -> float:
        """Compute retrieval recall against gold supporting-fact titles."""
        if not gold_titles:
            return 1.0

        retrieved_titles: set[str] = set()

        if hasattr(trace, "passages") and trace.passages:
            for results in trace.passages.values():
                for r in results:
                    retrieved_titles.add(r.title.lower().strip())

        for it in trace.iterations:
            for sr in it.get("sub_results", []):
                for t in sr.get("top_titles", []):
                    retrieved_titles.add(t.lower().strip())

        found = sum(
            1 for gt in gold_titles
            if gt.lower().strip() in retrieved_titles
        )
        return found / len(gold_titles)

    def _evaluate_single(
        self,
        question_data: dict,
        method: str,
    ) -> QuestionMetrics:
        """Run one question through the specified method and compute metrics."""
        question    = question_data["question"]
        gold_answer = question_data["answer"]
        qid         = question_data["id"]

        if method == "agentic":
            answer, trace = self.agentic.query(question)
        else:
            answer, trace = self.static.query(question)

        em  = exact_match(answer, gold_answer)
        f1  = token_f1(answer, gold_answer)
        rl  = rouge_l_score(answer, gold_answer)

        all_context = self._build_passage_context(trace)

        groundedness      = 0.0
        hallucination_rate = 1.0
        try:
            if method == "agentic" and trace.verification:
                groundedness      = trace.verification.get("confidence", 0.0)
                hallucination_rate = 1.0 - groundedness
            elif all_context.strip():
                vr = self.verifier.verify(answer, all_context)
                groundedness      = vr.confidence
                hallucination_rate = 1.0 - groundedness
        except Exception as e:
            log.warning("Groundedness check failed for %s: %s", qid, e)

        relevance = answer_relevance_llm(question, answer, counter=self.eval_counter)

        gold_titles   = list(question_data.get("supporting_facts", {}).keys())
        retrieval_rec = self._build_retrieval_recall(trace, gold_titles)

        error_cat  = classify_error(em, f1, retrieval_rec, groundedness)
        complexity = getattr(trace, "complexity", "medium")

        return QuestionMetrics(
            question_id=qid,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=answer,
            method=method,
            exact_match=em,
            f1=f1,
            rouge_l=rl,
            retrieval_recall=retrieval_rec,
            groundedness=groundedness,
            hallucination_rate=hallucination_rate,
            answer_relevance=relevance,
            latency_seconds=trace.elapsed_seconds,
            num_retrievals=trace.total_retrievals,
            num_llm_calls=trace.total_llm_calls,
            total_tokens=0,
            estimated_cost=0.0,
            error_category=error_cat,
            complexity=complexity,
        )

    def run(
        self,
        sample_size: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Run full comparative evaluation.

        Args:
            sample_size: Number of questions to evaluate.
            output_dir: Where to save results.

        Returns:
            Dict with "agentic" and "static" aggregate metrics.
        """
        sample_size = sample_size or cfg.eval_sample_size
        output_dir  = output_dir  or cfg.metrics_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        questions = load_eval_dataset(sample_size=sample_size)

        agentic_metrics: list[QuestionMetrics] = []
        static_metrics:  list[QuestionMetrics] = []

        for q in tqdm(questions, desc="Evaluating"):
            try:
                sm = self._evaluate_single(q, "static")
                static_metrics.append(sm)
            except Exception as e:
                log.error("Static eval failed for %s: %s", q["id"], e)

            try:
                am = self._evaluate_single(q, "agentic")
                agentic_metrics.append(am)
            except Exception as e:
                log.error("Agentic eval failed for %s: %s", q["id"], e)

        agentic_agg = compute_aggregates(agentic_metrics, "agentic")
        static_agg  = compute_aggregates(static_metrics,  "static")

        with open(output_dir / "agentic_per_question.json", "w") as f:
            json.dump([m.to_dict() for m in agentic_metrics], f, indent=2)
        with open(output_dir / "static_per_question.json", "w") as f:
            json.dump([m.to_dict() for m in static_metrics],  f, indent=2)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump({
                "agentic": agentic_agg.to_dict(),
                "static":  static_agg.to_dict(),
            }, f, indent=2)

        log.info("═══ RESULTS ═══")
        log.info(
            "Agentic — EM:%.3f | F1:%.3f | ROUGE-L:%.3f | Ground:%.3f"
            " | Relevance:%.3f | Latency:%.1fs",
            agentic_agg.avg_exact_match, agentic_agg.avg_f1,
            agentic_agg.avg_rouge_l,
            agentic_agg.avg_groundedness, agentic_agg.avg_answer_relevance,
            agentic_agg.avg_latency,
        )
        log.info(
            "Static  — EM:%.3f | F1:%.3f | ROUGE-L:%.3f | Ground:%.3f"
            " | Relevance:%.3f | Latency:%.1fs",
            static_agg.avg_exact_match, static_agg.avg_f1,
            static_agg.avg_rouge_l,
            static_agg.avg_groundedness, static_agg.avg_answer_relevance,
            static_agg.avg_latency,
        )

        return {
            "agentic": agentic_agg.to_dict(),
            "static":  static_agg.to_dict(),
        }
