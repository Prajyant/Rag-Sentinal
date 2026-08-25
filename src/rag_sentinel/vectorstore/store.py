"""Vector store: embedding storage, baseline statistics, and retrieval.

Uses ChromaDB as the default backend (easy setup, metadata filtering).
The baseline statistics (mean embedding, covariance) computed from the
clean corpus are stored here and consumed by the anomaly detector.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ..ingestion.chunker import Chunk
from .embed import EmbeddingModel


class BaselineStats:
    """Stores the clean-corpus embedding statistics for anomaly detection."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None          # (dim,)
        self.covariance: np.ndarray | None = None    # (dim, dim)
        self.inv_covariance: np.ndarray | None = None
        self.n_samples: int = 0

    def fit(self, embeddings: np.ndarray) -> None:
        """Compute mean and covariance from a (N, dim) embedding matrix."""
        self.n_samples = len(embeddings)
        self.mean = embeddings.mean(axis=0)
        # Use regularized covariance to avoid singularity
        cov = np.cov(embeddings, rowvar=False)
        reg = 1e-6 * np.eye(cov.shape[0])
        self.covariance = cov + reg
        self.inv_covariance = np.linalg.inv(self.covariance)
        logger.info(f"Baseline fitted on {self.n_samples} embeddings, dim={len(self.mean)}")

    def mahalanobis(self, vector: np.ndarray) -> float:
        """Return Mahalanobis distance from the baseline mean."""
        if self.mean is None:
            raise RuntimeError("Baseline not fitted — call fit() first")
        diff = vector - self.mean
        dist = float(np.sqrt(diff @ self.inv_covariance @ diff))
        return dist

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info(f"Baseline saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaselineStats":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info(f"Baseline loaded from {path} ({obj.n_samples} samples)")
        return obj


class VectorStore:
    """Thin wrapper around ChromaDB with convenience methods for RAG Sentinel.

    Responsibilities:
    - Store chunk embeddings + metadata
    - Maintain the clean-corpus baseline
    - Provide top-k similarity retrieval with optional risk-score filtering
    """

    BASELINE_COLLECTION = "baseline_corpus"
    DOCUMENTS_COLLECTION = "all_documents"

    def __init__(
        self,
        persist_path: str = "./chroma_db",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb is required: pip install chromadb") from exc

        self.persist_path = persist_path
        self.embedder = embedding_model or EmbeddingModel()

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=self.DOCUMENTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self.baseline = BaselineStats()
        self._baseline_path = Path(persist_path) / "baseline_stats.pkl"
        if self._baseline_path.exists():
            self.baseline = BaselineStats.load(self._baseline_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        is_baseline: bool = False,
    ) -> list[str]:
        """Embed and store chunks.  If is_baseline=True, refit the baseline stats."""
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress=True)

        ids, metas, docs = [], [], []
        for chunk, vec in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id or f"chunk_{id(chunk)}"
            ids.append(chunk_id)
            metas.append(
                {
                    k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                    for k, v in chunk.metadata.items()
                }
            )
            docs.append(chunk.content)

        # ChromaDB upsert in batches of 500
        batch = 500
        for start in range(0, len(ids), batch):
            self._collection.upsert(
                ids=ids[start : start + batch],
                embeddings=embeddings[start : start + batch].tolist(),
                documents=docs[start : start + batch],
                metadatas=metas[start : start + batch],
            )

        if is_baseline:
            self._refit_baseline(embeddings)

        logger.info(f"VectorStore: stored {len(chunks)} chunks (baseline={is_baseline})")
        return ids

    def _refit_baseline(self, new_embeddings: np.ndarray) -> None:
        """Accumulate new embeddings into the baseline stats and save."""
        # Retrieve existing baseline embeddings if any
        existing = self._client.get_or_create_collection(
            name=self.BASELINE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        # Store new ones
        ids = [f"bl_{i}" for i in range(len(new_embeddings))]
        existing.upsert(
            ids=ids,
            embeddings=new_embeddings.tolist(),
        )
        # Re-fetch all baseline embeddings and refit stats
        result = existing.get(include=["embeddings"])
        all_vecs = np.array(result["embeddings"], dtype=np.float32)
        self.baseline.fit(all_vecs)
        self.baseline.save(self._baseline_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_risk_score: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for a query, filtered by risk score.

        Returns a list of dicts with keys: id, content, metadata, distance.
        Chunks whose metadata risk_score > max_risk_score are excluded.
        """
        query_vec = self.embedder.encode_single(query)
        n_query = min(top_k * 4, self._collection.count())  # over-fetch to allow filtering
        if n_query == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_query,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            risk = float(meta.get("risk_score", 0.0))
            if risk > max_risk_score:
                continue
            hits.append(
                {"id": cid, "content": doc, "metadata": meta, "distance": dist}
            )
            if len(hits) >= top_k:
                break

        return hits

    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a single chunk."""
        result = self._collection.get(ids=[chunk_id], include=["embeddings"])
        if not result["embeddings"]:
            return None
        return np.array(result["embeddings"][0], dtype=np.float32)

    def update_risk_score(self, chunk_id: str, risk_score: float, decision: str) -> None:
        """Write the computed risk score back into the chunk's metadata."""
        self._collection.update(
            ids=[chunk_id],
            metadatas=[{"risk_score": str(risk_score), "decision": decision}],
        )

    @property
    def count(self) -> int:
        return self._collection.count()
