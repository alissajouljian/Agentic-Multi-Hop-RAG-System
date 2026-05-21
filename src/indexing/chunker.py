"""
Semantic chunker: splits documents into overlapping chunks at sentence
boundaries while respecting a maximum token budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm

from src.config import cfg
from src.indexing.corpus_loader import Document
from src.utils import get_logger, timer

log = get_logger(__name__)


@dataclass
class Chunk:
    """A retrieval unit derived from a parent document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int
    chunk_index: int  # position within the parent document

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _approx_token_count(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@timer
def chunk_documents(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Chunk]:
    """
    Split each Document into overlapping Chunks at sentence boundaries.

    The algorithm greedily accumulates sentences until the token budget is
    reached, then slides back by `chunk_overlap` tokens worth of sentences
    to start the next chunk.

    Args:
        documents: Source documents to chunk.
        chunk_size: Max tokens per chunk (default from config).
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        Flat list of Chunk objects.
    """
    chunk_size = chunk_size or cfg.chunk_size
    chunk_overlap = chunk_overlap or cfg.chunk_overlap

    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for doc in tqdm(documents, desc="Chunking"):
        if doc.sentences:
            sentences = doc.sentences
        else:
            sentences = _split_sentences(doc.text)

        if not sentences:
            continue

        current_sents: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _approx_token_count(sent)

            if current_tokens + sent_tokens > chunk_size and current_sents:
                chunk_text = " ".join(current_sents)
                all_chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_counter:07d}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=chunk_text,
                        token_count=current_tokens,
                        chunk_index=chunk_counter,
                    )
                )
                chunk_counter += 1

                overlap_tokens = 0
                keep_from = len(current_sents)
                for i in range(len(current_sents) - 1, -1, -1):
                    overlap_tokens += _approx_token_count(current_sents[i])
                    if overlap_tokens >= chunk_overlap:
                        keep_from = i
                        break
                current_sents = current_sents[keep_from:]
                current_tokens = sum(
                    _approx_token_count(s) for s in current_sents
                )

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            chunk_text = " ".join(current_sents)
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{chunk_counter:07d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    token_count=current_tokens,
                    chunk_index=chunk_counter,
                )
            )
            chunk_counter += 1

    log.info(
        "Created %d chunks from %d documents (avg %.0f tokens/chunk)",
        len(all_chunks),
        len(documents),
        sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1),
    )
    return all_chunks
