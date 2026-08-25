"""Cross-reference / consistency checker (inspired by TrustRAG / RAGDefender).

A poisoned chunk that contradicts the retrieval consensus is a signal
that's independent from the embedding-anomaly and content classifiers —
making the ensemble more robust.

Two sub-checks:
  1. Pairwise semantic agreement:  a chunk that disagrees with most of
     the other retrieved chunks scores high.
  2. Self-consistency (optional):  ask the LLM whether a chunk actually
     supports the answer it's about to give; flag contradictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class ConsistencyChecker:
    """Scores each retrieved chunk for coherence with the retrieval set.

    Parameters
    ----------
    embedding_model:
        The same EmbeddingModel instance used by the vector store (passed in
        at construction to avoid loading the model twice).
    agreement_threshold:
        Mean pairwise cosine similarity below which a chunk is flagged.
    """

    def __init__(
        self,
        embedding_model=None,
        agreement_threshold: float = 0.30,
    ) -> None:
        self._embedder = embedding_model
        self.agreement_threshold = agreement_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        pre_computed_embeddings: np.ndarray | None = None,
    ) -> list[float]:
        """Return one score per chunk in [0, 1] (higher = more suspicious).

        Parameters
        ----------
        chunks:
            List of chunk dicts from VectorStore.retrieve(), each with a
            "content" key.
        pre_computed_embeddings:
            Optional (N, dim) float32 array — avoids re-embedding if already
            available.
        """
        if len(chunks) <= 1:
            return [0.0] * len(chunks)

        texts = [c["content"] for c in chunks]

        if pre_computed_embeddings is not None:
            vecs = pre_computed_embeddings
        elif self._embedder is not None:
            vecs = self._embedder.encode(texts)
        else:
            logger.warning("ConsistencyChecker has no embedding model — returning zeros")
            return [0.0] * len(chunks)

        # Cosine similarity matrix  (N, N)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = vecs / norms
        sim_matrix = normed @ normed.T   # (N, N), values in [-1, 1]

        scores: list[float] = []
        n = len(chunks)

        for i in range(n):
            # Mean similarity of chunk i to all *other* chunks
            others = [sim_matrix[i, j] for j in range(n) if j != i]
            mean_agreement = float(np.mean(others))

            # Score: how much below the agreement threshold is this chunk?
            # Threshold 0.30 — chunks below this have low semantic overlap with peers
            if mean_agreement < self.agreement_threshold:
                # Scale from 0 (at threshold) to 1 (at -1 = perfect disagreement)
                score = (self.agreement_threshold - mean_agreement) / (
                    self.agreement_threshold + 1.0
                )
            else:
                score = 0.0

            scores.append(float(score))

        logger.debug(f"Consistency scores: {[round(s, 3) for s in scores]}")
        return scores

    def llm_self_consistency_score(
        self,
        chunk_content: str,
        query: str,
        answer: str,
        llm_fn: Any,  # callable(prompt: str) -> str
    ) -> float:
        """Ask the LLM whether this chunk actually supports the given answer.

        Returns 0.0 if the chunk is consistent, up to 1.0 if it contradicts.
        This is the "self-consistency check" from the todo list — optional
        because it adds one LLM call per chunk per query.

        Parameters
        ----------
        llm_fn:
            A callable that takes a prompt string and returns the LLM's
            text response.  Provided by the pipeline orchestrator.
        """
        prompt = (
            f"You are a document verification assistant. "
            f"Given the following retrieved passage, determine whether it "
            f"directly supports the answer provided.\n\n"
            f"Question: {query}\n\n"
            f"Retrieved passage:\n{chunk_content}\n\n"
            f"Answer: {answer}\n\n"
            f"Does the passage support the answer? Reply with only "
            f"'SUPPORTS', 'NEUTRAL', or 'CONTRADICTS'."
        )
        try:
            response = llm_fn(prompt).strip().upper()
            if "CONTRADICTS" in response:
                return 1.0
            if "NEUTRAL" in response:
                return 0.4
            return 0.0  # SUPPORTS
        except Exception as exc:
            logger.warning(f"LLM self-consistency check failed: {exc}")
            return 0.0
