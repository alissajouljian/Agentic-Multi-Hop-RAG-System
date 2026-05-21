"""
Shared utilities: logging, timing, text helpers.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)-25s %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def timer(func: Callable) -> Callable:
    """Log execution time of the decorated function."""
    log = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info("⏱  %s finished in %.2f s", func.__qualname__, elapsed)
        return result

    return wrapper


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # ASCII printable + newline
    return text.strip()


def normalize_answer(answer: str) -> str:
    """Lower-case, strip articles/punctuation for EM/F1 scoring."""
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = " ".join(answer.split())
    return answer


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer after normalization."""
    return normalize_answer(text).split()


class TokenCounter:
    """Lightweight token & cost tracker for OpenAI calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.calls: int = 0

    def update(self, usage: dict) -> None:
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost_usd(self, prompt_rate: float = 0.15e-6,
                           completion_rate: float = 0.60e-6) -> float:
        """Estimate cost using GPT-4o-mini per-token pricing."""
        return (self.prompt_tokens * prompt_rate
                + self.completion_tokens * completion_rate)

    def summary(self) -> str:
        return (
            f"LLM calls: {self.calls} | "
            f"Tokens: {self.total_tokens:,} "
            f"(prompt {self.prompt_tokens:,}, completion {self.completion_tokens:,}) | "
            f"Est. cost: ${self.estimated_cost_usd():.4f}"
        )
