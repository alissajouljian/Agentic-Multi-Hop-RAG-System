"""
Evaluation metrics for RAG systems.

Implements:
  - Exact Match (EM)
  - Token-level F1
  - ROUGE-L
  - Retrieval Recall@K
  - Groundedness (LLM claim-level)
  - Hallucination Rate
  - Answer Relevance (LLM-as-judge)
  - Latency and cost tracking
  - Error taxonomy classification
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.config import cfg
from src.retrieval.retriever import RetrievalResult
from src.utils import get_logger, normalize_answer, tokenize, TokenCounter

log = get_logger(__name__)


@dataclass
class QuestionMetrics:
    """Metrics for a single question."""

    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    method: str  # "agentic" or "static"

    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    retrieval_recall: float = 0.0
    groundedness: float = 0.0
    hallucination_rate: float = 0.0
    answer_relevance: float = 0.0
    latency_seconds: float = 0.0
    num_retrievals: int = 0
    num_llm_calls: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    complexity: str = "medium"
    error_category: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AggregateMetrics:
    """Aggregated metrics over the entire evaluation set."""

    method: str
    num_questions: int = 0
    avg_exact_match: float = 0.0
    avg_f1: float = 0.0
    avg_rouge_l: float = 0.0
    avg_retrieval_recall: float = 0.0
    avg_groundedness: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_latency: float = 0.0
    avg_retrievals: float = 0.0
    avg_llm_calls: float = 0.0
    total_cost: float = 0.0
    error_distribution: dict = field(default_factory=dict)
    complexity_distribution: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def exact_match(prediction: str, gold: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, gold: str) -> float:
    """ROUGE-L F1: longest common subsequence based score."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall    = lcs / n
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(
    retrieved: list[RetrievalResult],
    gold_titles: list[str],
) -> float:
    """
    Fraction of gold supporting-fact titles that appear in retrieved results.
    """
    if not gold_titles:
        return 1.0

    retrieved_titles = {r.title.lower().strip() for r in retrieved}
    gold_set = {t.lower().strip() for t in gold_titles}

    found = len(gold_set & retrieved_titles)
    return found / len(gold_set)


def answer_relevance_llm(
    question: str,
    answer: str,
    client: Optional[OpenAI] = None,
    counter: Optional[TokenCounter] = None,
) -> float:
    """
    LLM-as-judge: rate how relevant the answer is to the question (0–1).
    """
    client  = client  or OpenAI(api_key=cfg.openai_api_key)
    counter = counter or TokenCounter()

    prompt = (
        "Rate how well the following answer addresses the question. "
        "Return ONLY a number between 0.0 and 1.0.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Score:"
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        counter.update({
            "prompt_tokens":      resp.usage.prompt_tokens,
            "completion_tokens":  resp.usage.completion_tokens,
        })
        score = float(resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))

    except Exception as e:
        log.warning("Answer relevance scoring failed: %s", e)
        return 0.0


def classify_error(
    em: float,
    f1: float,
    retrieval_rec: float,
    groundedness: float,
) -> str:
    """Categorise the error type for analysis."""
    if em >= 1.0 or f1 >= 0.4:
        return "correct"
    if retrieval_rec < 0.4:
        return "retrieval_failure"
    if groundedness < 0.5:
        return "hallucination"
    return "reasoning_failure"


def compute_aggregates(
    metrics_list: list[QuestionMetrics],
    method: str,
) -> AggregateMetrics:
    """Compute aggregate statistics from per-question metrics."""
    n = len(metrics_list)
    if n == 0:
        return AggregateMetrics(method=method)

    errors: dict[str, int] = {}
    complexities: dict[str, int] = {}
    for m in metrics_list:
        errors[m.error_category] = errors.get(m.error_category, 0) + 1
        complexities[m.complexity] = complexities.get(m.complexity, 0) + 1

    return AggregateMetrics(
        method=method,
        num_questions=n,
        avg_exact_match=sum(m.exact_match for m in metrics_list) / n,
        avg_f1=sum(m.f1 for m in metrics_list) / n,
        avg_rouge_l=sum(m.rouge_l for m in metrics_list) / n,
        avg_retrieval_recall=sum(m.retrieval_recall for m in metrics_list) / n,
        avg_groundedness=sum(m.groundedness for m in metrics_list) / n,
        avg_hallucination_rate=sum(m.hallucination_rate for m in metrics_list) / n,
        avg_answer_relevance=sum(m.answer_relevance for m in metrics_list) / n,
        avg_latency=sum(m.latency_seconds for m in metrics_list) / n,
        avg_retrievals=sum(m.num_retrievals for m in metrics_list) / n,
        avg_llm_calls=sum(m.num_llm_calls for m in metrics_list) / n,
        total_cost=sum(m.estimated_cost for m in metrics_list),
        error_distribution=errors,
        complexity_distribution=complexities,
    )
