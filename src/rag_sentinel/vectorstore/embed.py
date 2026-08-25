"""Embedding model wrapper.

Benchmarks three models at runtime and caches the chosen one.
The default is all-MiniLM-L6-v2 (fast, decent quality).
For best accuracy swap to bge-large-en-v1.5.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from loguru import logger


class EmbeddingModel:
    """Wraps sentence-transformers for batch embedding.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Recommended options:
            - "all-MiniLM-L6-v2"     fast, 384-dim
            - "BAAI/bge-small-en-v1.5"  balanced, 384-dim
            - "BAAI/bge-large-en-v1.5"  best quality, 1024-dim (slower)
    device:
        "cpu", "cuda", or "mps" — auto-detected if None.
    batch_size:
        Number of texts encoded in one forward pass.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.info(f"Loading embedding model {model_name!r} on {device}")
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dimension: {self.dim}")

    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of strings → (N, dim) float32 array."""
        if isinstance(texts, str):
            texts = [texts]

        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience method — returns a (dim,) vector."""
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Benchmark helper (Phase 3: compare models before committing)
    # ------------------------------------------------------------------

    @staticmethod
    def benchmark(
        texts: list[str],
        model_names: list[str] | None = None,
    ) -> dict[str, dict]:
        """Time and compare multiple embedding models on the same texts.

        Returns a dict of {model_name: {latency_ms, dim}}.
        """
        if model_names is None:
            model_names = [
                "all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
            ]

        results: dict[str, dict] = {}
        for name in model_names:
            try:
                model = EmbeddingModel(model_name=name)
                t0 = time.perf_counter()
                model.encode(texts)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results[name] = {"latency_ms": round(elapsed_ms, 1), "dim": model.dim}
                logger.info(f"  {name}: {elapsed_ms:.0f} ms, dim={model.dim}")
            except Exception as exc:
                results[name] = {"error": str(exc)}

        return results
