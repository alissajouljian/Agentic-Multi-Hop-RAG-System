"""
Evaluation dataset loader for HotpotQA questions with gold answers.
"""

from __future__ import annotations

from typing import Optional

from src.config import cfg
from src.indexing.corpus_loader import load_hotpotqa_questions
from src.utils import get_logger

log = get_logger(__name__)


def load_eval_dataset(
    sample_size: Optional[int] = None,
    split: Optional[str] = None,
) -> list[dict]:
    """
    Load evaluation questions with gold answers and supporting facts.

    Args:
        sample_size: Number of questions to load (default from config).
        split: Dataset split (default from config).

    Returns:
        List of question dicts with keys:
            id, question, answer, type, level, supporting_facts, context
    """
    sample_size = sample_size or cfg.eval_sample_size
    split = split or cfg.eval_split

    questions = load_hotpotqa_questions(split=split, sample_size=sample_size)

    types = {}
    levels = {}
    for q in questions:
        types[q["type"]] = types.get(q["type"], 0) + 1
        levels[q["level"]] = levels.get(q["level"], 0) + 1

    log.info("Eval dataset: %d questions", len(questions))
    log.info("  Types:  %s", types)
    log.info("  Levels: %s", levels)

    return questions
