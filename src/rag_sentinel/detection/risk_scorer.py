"""Risk score combiner — the final decision layer.

Combines outputs from the three detectors into a single risk score and
emits a Pass / Flag / Quarantine decision.

Two modes:
  1. Weighted sum (default until the meta-classifier is trained)
  2. Stacked meta-classifier (logistic regression / shallow MLP trained on
     the four detector outputs as features, with Platt-scaled probabilities)

The meta-classifier mode is enabled once enough labelled data is collected
from the human-review feedback loop on "Flag" items.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class Decision(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    QUARANTINE = "quarantine"


@dataclass
class DetectorOutputs:
    """Structured container for the four detector scores."""
    anomaly_score: float       # [0, 1] from AnomalyDetector
    rule_score: float          # [0, 1] from RuleScreener
    classifier_score: float    # [0, 1] from ContentClassifier
    consistency_score: float   # [0, 1] from ConsistencyChecker
    matched_rules: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    """Full decision record logged per chunk per query."""
    chunk_id: str
    risk_score: float
    decision: Decision
    detector_outputs: DetectorOutputs
    query: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "risk_score": round(self.risk_score, 4),
            "decision": self.decision.value,
            "anomaly_score": round(self.detector_outputs.anomaly_score, 4),
            "rule_score": round(self.detector_outputs.rule_score, 4),
            "classifier_score": round(self.detector_outputs.classifier_score, 4),
            "consistency_score": round(self.detector_outputs.consistency_score, 4),
            "matched_rules": self.detector_outputs.matched_rules,
            "query": self.query,
            "explanation": self.explanation,
        }


class RiskScorer:
    """Combines detector scores → final risk score + decision.

    Parameters
    ----------
    weights:
        Dict of detector_name → weight for the weighted-sum mode.
        Must sum to 1.0.
    pass_threshold:
        Risk score below this → PASS.
    flag_threshold:
        Risk score above this → QUARANTINE; between pass and flag → FLAG.
    use_meta_classifier:
        If True and a trained meta-classifier is loaded, use it instead of
        the weighted sum.
    """

    DEFAULT_WEIGHTS = {
        "anomaly": 0.35,
        "rule": 0.25,
        "classifier": 0.30,
        "consistency": 0.10,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        pass_threshold: float = 0.30,
        flag_threshold: float = 0.70,
        use_meta_classifier: bool = False,
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.pass_threshold = pass_threshold
        self.flag_threshold = flag_threshold
        self.use_meta_classifier = use_meta_classifier
        self._meta_clf = None

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def score(
        self,
        outputs: DetectorOutputs,
        chunk_id: str = "",
        query: str = "",
    ) -> RiskDecision:
        """Combine detector outputs into a single RiskDecision."""
        if self.use_meta_classifier and self._meta_clf is not None:
            risk = self._meta_score(outputs)
        else:
            risk = self._weighted_sum(outputs)

        decision = self._threshold(risk)
        explanation = self._explain(outputs, risk, decision)

        logger.debug(f"Chunk {chunk_id!r}: risk={risk:.3f} → {decision.value}")
        return RiskDecision(
            chunk_id=chunk_id,
            risk_score=risk,
            decision=decision,
            detector_outputs=outputs,
            query=query,
            explanation=explanation,
        )

    def _weighted_sum(self, o: DetectorOutputs) -> float:
        w = self.weights
        raw = (
            w["anomaly"] * o.anomaly_score
            + w["rule"] * o.rule_score
            + w["classifier"] * o.classifier_score
            + w["consistency"] * o.consistency_score
        )
        return float(min(max(raw, 0.0), 1.0))

    def _meta_score(self, o: DetectorOutputs) -> float:
        X = np.array([[
            o.anomaly_score,
            o.rule_score,
            o.classifier_score,
            o.consistency_score,
        ]])
        return float(self._meta_clf.predict_proba(X)[0, 1])

    def _threshold(self, risk: float) -> Decision:
        if risk >= self.flag_threshold:
            return Decision.QUARANTINE
        if risk >= self.pass_threshold:
            return Decision.FLAG
        return Decision.PASS

    @staticmethod
    def _explain(o: DetectorOutputs, risk: float, decision: Decision) -> str:
        parts = []
        if o.anomaly_score > 0.5:
            parts.append(f"embedding anomaly ({o.anomaly_score:.2f})")
        if o.rule_score > 0.3:
            parts.append(f"rule matches: {', '.join(o.matched_rules[:3])}")
        if o.classifier_score > 0.5:
            parts.append(f"classifier ({o.classifier_score:.2f})")
        if o.consistency_score > 0.4:
            parts.append(f"consistency ({o.consistency_score:.2f})")
        if not parts:
            return f"risk={risk:.2f} → {decision.value}"
        return f"risk={risk:.2f} → {decision.value} | triggers: {'; '.join(parts)}"

    # ------------------------------------------------------------------
    # Meta-classifier training
    # ------------------------------------------------------------------

    def train_meta_classifier(
        self,
        detector_outputs: list[DetectorOutputs],
        labels: list[int],  # 0 = clean, 1 = malicious
    ) -> dict[str, float]:
        """Train a stacked meta-classifier on detector output features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import cross_validate

        X = np.array([
            [o.anomaly_score, o.rule_score, o.classifier_score, o.consistency_score]
            for o in detector_outputs
        ])
        y = np.array(labels)

        base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
        calibrated.fit(X, y)
        self._meta_clf = calibrated
        self.use_meta_classifier = True

        cv = cross_validate(
            base, X, y,
            cv=5,
            scoring=["precision", "recall", "f1", "roc_auc"],
        )
        metrics = {k: float(v.mean()) for k, v in cv.items() if k.startswith("test_")}
        logger.info(f"Meta-classifier CV: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "RiskScorer":
        with open(path, "rb") as fh:
            return pickle.load(fh)
