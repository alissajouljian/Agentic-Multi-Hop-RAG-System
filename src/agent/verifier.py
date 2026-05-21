"""
Self-Verification module: uses LLM to extract and verify claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from src.config import cfg
from src.utils import get_logger, TokenCounter

log = get_logger(__name__)


@dataclass
class VerificationResult:
    confidence: float
    total_claims: int
    supported_claims: int
    unsupported: list[str]
    details: list[dict]
    passed: bool


class Verifier:
    """
    LLM-based self-verification of generated answers.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.client = client or OpenAI(api_key=cfg.openai_api_key)
        self.counter = token_counter or TokenCounter()

    def _extract_claims(self, answer: str) -> list[str]:
        prompt = (
            "Extract atomic factual claims from the answer. "
            "Return each claim on a new line starting with '- '.\n\n"
            f"Answer: {answer}\n\nClaims:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            return [line.lstrip("- ").strip() for line in raw.split("\n") if line.strip().startswith("-")]
        except:
            return [answer]

    def verify(
        self,
        answer: str,
        context: str,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        threshold = threshold or cfg.verification_threshold
        claims = self._extract_claims(answer)
        
        details = []
        supported = 0
        unsupported_claims = []

        for claim in claims:
            prompt = (
                f"Context: {context[:4000]}\n\n"
                f"Claim: {claim}\n\n"
                "Does the context support the claim? Answer only 'YES' or 'NO'."
            )
            try:
                resp = self.client.chat.completions.create(
                    model=cfg.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                label = resp.choices[0].message.content.strip().upper()
                is_supported = "YES" in label
                
                details.append({"claim": claim, "supported": is_supported})
                if is_supported:
                    supported += 1
                else:
                    unsupported_claims.append(claim)
            except:
                details.append({"claim": claim, "supported": False})
                unsupported_claims.append(claim)

        confidence = supported / max(len(claims), 1)
        return VerificationResult(
            confidence=confidence,
            total_claims=len(claims),
            supported_claims=supported,
            unsupported=unsupported_claims,
            details=details,
            passed=confidence >= threshold,
        )
