"""
Indexer: builds BM25 and FAISS (dense) indices from chunks, persists
them to disk, and exposes a unified hybrid search interface.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.config import cfg
from src.indexing.chunker import Chunk
from src.utils import get_logger, timer

log = get_logger(__name__)


class ChunkStore:
    """Persisted chunk metadata store backed by JSONL."""

    def __init__(self, chunks: Optional[list[Chunk]] = None) -> None:
        self._map: dict[str, Chunk] = {}
        self._ordered: list[Chunk] = []
        if chunks:
            for c in chunks:
                self._map[c.chunk_id] = c
                self._ordered.append(c)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._map.get(chunk_id)

    def get_by_index(self, idx: int) -> Chunk:
        return self._ordered[idx]

    def __len__(self) -> int:
        return len(self._ordered)

    def texts(self) -> list[str]:
        return [c.text for c in self._ordered]

    def titles(self) -> list[str]:
        return [c.title for c in self._ordered]

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            for c in self._ordered:
                f.write(json.dumps(c.to_dict()) + "\n")
        log.info("Saved %d chunks to %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        chunks = []
        with open(path) as f:
            for line in f:
                chunks.append(Chunk.from_dict(json.loads(line)))
        store = cls(chunks)
        log.info("Loaded %d chunks from %s", len(store), path)
        return store


class BM25Index:
    """Sparse BM25 index over chunk texts."""

    def __init__(self) -> None:
        self._index = None
        self._tokenized: list[list[str]] = []

    @timer
    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        log.info("Building BM25 index over %d chunks …", len(chunks))
        self._tokenized = [c.text.lower().split() for c in chunks]
        self._index = BM25Okapi(self._tokenized)
        log.info("BM25 index built")

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Return list of (chunk_index, score) sorted desc."""
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "tokenized": self._tokenized}, f)
        log.info("BM25 index saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._index = data["index"]
        obj._tokenized = data["tokenized"]
        log.info("BM25 index loaded from %s", path)
        return obj


class DenseIndex:
    """Dense embedding index backed by FAISS."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or cfg.embedding_model
        self._index = None
        self._model = None
        self._dim: int = 0

    def _load_model(self):
        if self._model is None:
            if "text-embedding" in self._model_name:
                from openai import OpenAI
                log.info("Using OpenAI embedding model: %s", self._model_name)
                self._model = OpenAI(api_key=cfg.openai_api_key)
            else:
                from sentence_transformers import SentenceTransformer
                log.info("Loading local embedding model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)

    @timer
    def build(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        import faiss

        self._load_model()
        texts = [f"{c.title}: {c.text}" for c in chunks]

        if "text-embedding" in self._model_name:
            log.info("Encoding %d chunks with OpenAI %s …", len(texts), self._model_name)
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._model.embeddings.create(input=batch, model=self._model_name)
                embeddings_list.extend([e.embedding for e in resp.data])
            embeddings = np.array(embeddings_list, dtype="float32")
        else:
            log.info("Encoding %d chunks with %s (CPU) …", len(texts), self._model_name)
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                device="cpu",
            )
            embeddings = np.array(embeddings, dtype="float32")
        self._dim = embeddings.shape[1]

        log.info("Building FAISS index (dim=%d) …", self._dim)
        self._index = faiss.IndexFlatIP(self._dim)  # inner-product on L2-normed vecs
        self._index.add(embeddings)
        log.info("FAISS index built with %d vectors", self._index.ntotal)

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Return list of (chunk_index, score) sorted desc."""
        self._load_model()
        
        if "text-embedding" in self._model_name:
            resp = self._model.embeddings.create(input=[query], model=self._model_name)
            qvec = np.array([resp.data[0].embedding], dtype="float32")
        else:
            qvec = self._model.encode(
                [query], normalize_embeddings=True
            ).astype("float32")
            
        scores, indices = self._index.search(qvec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self._index, str(path))
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"dim": self._dim, "model": self._model_name}, f)
        log.info("Dense index saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        import faiss

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        obj = cls(model_name=meta["model"])
        obj._dim = meta["dim"]
        obj._index = faiss.read_index(str(path))
        log.info("Dense index loaded from %s (%d vectors)", path, obj._index.ntotal)
        return obj


class HybridIndex:
    """Reciprocal Rank Fusion of BM25 and Dense retrieval."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        chunk_store: ChunkStore,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.chunk_store = chunk_store

    def search(
        self,
        query: str,
        top_k: int = 20,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion of BM25 and dense results.

        RRF score = Σ  weight / (rrf_k + rank)

        Returns:
            List of (Chunk, fused_score) sorted by descending score.
        """
        bm25_results = self.bm25.search(query, top_k=cfg.bm25_top_k)
        dense_results = self.dense.search(query, top_k=cfg.dense_top_k)

        scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0.0) + bm25_weight / (rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0.0) + dense_weight / (rrf_k + rank + 1)

        sorted_indices = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

        return [
            (self.chunk_store.get_by_index(idx), scores[idx])
            for idx in sorted_indices
        ]

    @classmethod
    @timer
    def build_all(cls, chunks: list[Chunk]) -> "HybridIndex":
        """Build BM25 + Dense indices and chunk store from scratch."""
        store = ChunkStore(chunks)
        bm25 = BM25Index()
        bm25.build(chunks)
        dense = DenseIndex()
        dense.build(chunks)
        return cls(bm25=bm25, dense=dense, chunk_store=store)

    def save_all(self, index_dir: Optional[Path] = None) -> None:
        index_dir = index_dir or cfg.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_store.save(index_dir / "chunks.jsonl")
        self.bm25.save(index_dir / "bm25.pkl")
        self.dense.save(index_dir / "dense.faiss")
        log.info("All indices saved to %s", index_dir)

    @classmethod
    def load_all(cls, index_dir: Optional[Path] = None) -> "HybridIndex":
        index_dir = index_dir or cfg.index_dir
        store = ChunkStore.load(index_dir / "chunks.jsonl")
        bm25 = BM25Index.load(index_dir / "bm25.pkl")
        dense = DenseIndex.load(index_dir / "dense.faiss")
        return cls(bm25=bm25, dense=dense, chunk_store=store)
