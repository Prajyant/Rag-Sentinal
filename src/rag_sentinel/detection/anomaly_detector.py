"""Embedding-space anomaly detector.

Supports three methods, benchmarked against the clean baseline:
  1. mahalanobis  — Mahalanobis distance in whitened embedding space (default)
  2. isolation_forest — sklearn IsolationForest (good baseline)
  3. lof          — Local Outlier Factor

The detector outputs a probability-like score in [0, 1] where higher = more anomalous.
Scores are calibrated via percentile normalization over the validation set.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from ..vectorstore.store import BaselineStats


Method = Literal["mahalanobis", "isolation_forest", "lof"]


class AnomalyDetector:
    """Detects embedding-space outliers relative to a clean corpus baseline.

    Parameters
    ----------
    method:
        Detection algorithm.  Start with "mahalanobis"; train IsolationForest
        as a comparison baseline for the ablation study.
    score_percentile_cap:
        Raw Mahalanobis distances are normalized to [0, 1] by capping at this
        percentile of the validation-set distribution.
    """

    def __init__(
        self,
        method: Method = "mahalanobis",
        score_percentile_cap: float = 99.0,
    ) -> None:
        self.method = method
        self.score_percentile_cap = score_percentile_cap
        self._baseline: BaselineStats | None = None
        self._cap_value: float | None = None   # set during calibrate()
        self._sklearn_model = None             # IsolationForest / LOF

    # ------------------------------------------------------------------
    # Fitting / calibration
    # ------------------------------------------------------------------

    def fit(self, baseline: BaselineStats) -> None:
        """Attach a fitted BaselineStats object (Mahalanobis path)."""
        self._baseline = baseline
        logger.info(f"AnomalyDetector[{self.method}]: baseline attached ({baseline.n_samples} samples)")

    def fit_sklearn(self, embeddings: np.ndarray) -> None:
        """Fit IsolationForest or LOF on clean embeddings."""
        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self._sklearn_model = IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42,
                n_jobs=-1,
            )
            self._sklearn_model.fit(embeddings)
            logger.info(f"IsolationForest fitted on {len(embeddings)} samples")

        elif self.method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            self._sklearn_model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,
                novelty=True,  # enables predict on new samples
                n_jobs=-1,
            )
            self._sklearn_model.fit(embeddings)
            logger.info(f"LOF fitted on {len(embeddings)} samples")

    def calibrate(self, validation_embeddings: np.ndarray) -> None:
        """Set the normalization cap using the validation set distribution.

        After calibration, scores from clean validation docs cluster near 0
        and outliers push toward 1.
        """
        raw_scores = np.array([self._raw_score(v) for v in validation_embeddings])
        self._cap_value = float(np.percentile(raw_scores, self.score_percentile_cap))
        logger.info(
            f"AnomalyDetector calibrated: {self.score_percentile_cap}th-pct cap = {self._cap_value:.4f}"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, embedding: np.ndarray) -> float:
        """Return anomaly score in [0, 1] for a single embedding."""
        raw = self._raw_score(embedding)
        if self._cap_value and self._cap_value > 0:
            return float(min(raw / self._cap_value, 1.0))
        return float(min(raw / 100.0, 1.0))  # fallback normalization

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorized scoring for a batch of embeddings."""
        return np.array([self.score(v) for v in embeddings], dtype=np.float32)

    def _raw_score(self, embedding: np.ndarray) -> float:
        """Un-normalized anomaly score."""
        if self.method == "mahalanobis":
            if self._baseline is None:
                raise RuntimeError("Call fit() before scoring")
            return self._baseline.mahalanobis(embedding)

        elif self.method in ("isolation_forest", "lof"):
            if self._sklearn_model is None:
                raise RuntimeError("Call fit_sklearn() before scoring")
            # sklearn returns negative scores; flip so higher = more anomalous
            raw = -float(self._sklearn_model.score_samples([embedding])[0])
            return max(raw, 0.0)

        raise ValueError(f"Unknown method: {self.method!r}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        with open(path, "rb") as fh:
            return pickle.load(fh)
