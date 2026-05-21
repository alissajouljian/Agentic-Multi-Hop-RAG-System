"""
Corpus loader for HotpotQA Wikipedia paragraphs.

Downloads the HotpotQA dataset from HuggingFace, extracts the Wikipedia
paragraph contexts, and de-duplicates them into a flat list of Document
objects ready for chunking and indexing.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import cfg
from src.utils import get_logger, timer, clean_text

log = get_logger(__name__)


@dataclass
class Document:
    """A single paragraph / passage from the corpus."""

    doc_id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    source: str = "hotpotqa-wiki"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)


def _make_doc_id(title: str, text: str) -> str:
    """Deterministic ID from content hash."""
    h = hashlib.md5(f"{title}::{text}".encode()).hexdigest()[:12]
    return f"doc_{h}"


@timer
def load_hotpotqa_corpus(
    subset_size: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> list[Document]:
    """
    Load Wikipedia paragraphs from HotpotQA dataset.

    Each HotpotQA example has a 'context' field containing a list of
    (title, sentence_list) tuples. We flatten and de-duplicate these into
    Document objects.

    Args:
        subset_size: Max number of unique documents to return (None = all).
        cache_path: If provided, load from / save to this JSON-lines file.

    Returns:
        List of Document objects.
    """
    if cache_path is None:
        cache_path = cfg.raw_dir / "corpus.jsonl"

    if cache_path.exists():
        log.info("Loading cached corpus from %s", cache_path)
        docs = []
        with open(cache_path, "r") as f:
            for line in f:
                docs.append(Document.from_dict(json.loads(line)))
                if subset_size and len(docs) >= subset_size:
                    break
        log.info("Loaded %d documents from cache", len(docs))
        return docs

    log.info("Downloading HotpotQA dataset from HuggingFace …")
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

    seen: set[str] = set()
    docs: list[Document] = []

    for split_name in ["train", "validation"]:
        if split_name not in ds:
            continue
        log.info("Processing split: %s (%d examples)", split_name, len(ds[split_name]))
        for example in tqdm(ds[split_name], desc=f"Extracting [{split_name}]"):
            context = example.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])

            for title, sentences in zip(titles, sentences_list):
                full_text = " ".join(sentences)
                doc_id = _make_doc_id(title, full_text)
                if doc_id in seen:
                    continue
                seen.add(doc_id)

                docs.append(
                    Document(
                        doc_id=doc_id,
                        title=clean_text(title),
                        text=clean_text(full_text),
                        sentences=[clean_text(s) for s in sentences],
                    )
                )

                if subset_size and len(docs) >= subset_size:
                    break
            if subset_size and len(docs) >= subset_size:
                break
        if subset_size and len(docs) >= subset_size:
            break

    log.info("Extracted %d unique documents", len(docs))

    cfg.ensure_dirs()
    with open(cache_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict()) + "\n")
    log.info("Cached corpus to %s", cache_path)

    return docs


@timer
def load_hotpotqa_questions(
    split: str = "validation",
    sample_size: Optional[int] = None,
) -> list[dict]:
    """
    Load HotpotQA questions with gold answers and supporting facts.

    Returns list of dicts with keys:
        - question, answer, type, level
        - supporting_facts: {title: [sent_idx, ...]}
        - context: {title: [sentences]}
    """
    from datasets import load_dataset

    log.info("Loading HotpotQA questions [%s] …", split)
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)

    questions = []
    for ex in tqdm(ds, desc="Loading questions"):
        sf_titles = ex["supporting_facts"]["title"]
        sf_ids = ex["supporting_facts"]["sent_id"]
        sup_facts: dict[str, list[int]] = {}
        for t, sid in zip(sf_titles, sf_ids):
            sup_facts.setdefault(t, []).append(sid)

        ctx_titles = ex["context"]["title"]
        ctx_sents = ex["context"]["sentences"]
        context: dict[str, list[str]] = {}
        for t, sents in zip(ctx_titles, ctx_sents):
            context[t] = sents

        questions.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "supporting_facts": sup_facts,
                "context": context,
            }
        )
        if sample_size and len(questions) >= sample_size:
            break

    log.info("Loaded %d questions", len(questions))
    return questions
