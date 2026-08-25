"""Evaluation script — Phase 6.

Runs the full test set through the pipeline and computes:
  - Precision, Recall, F1, FPR — overall and per attack category
  - AUC-ROC and AUC-PR
  - Ablation study (each detector alone vs. combinations)
  - Latency percentiles (p50/p95/p99)
  - Bootstrap confidence intervals on F1/AUC

Usage:
  python -m rag_sentinel.eval.run_eval \\
      --labels data/labels.csv \\
      --output eval/results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..detection import AnomalyDetector, ContentClassifier, RuleScreener
from ..detection.consistency_checker import ConsistencyChecker
from ..detection.risk_scorer import DetectorOutputs, RiskScorer
from ..pipeline import RAGSentinelPipeline


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    lower = float(np.percentile(scores, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(scores, (1 + confidence) / 2 * 100))
    return lower, upper


def evaluate(
    pipeline: RAGSentinelPipeline,
    labels_df: pd.DataFrame,
) -> dict:
    """Run evaluation on a DataFrame with columns: text, label (0/1), attack_category.

    Returns a nested results dict.
    """
    texts = labels_df["text"].tolist()
    y_true = labels_df["label"].values
    categories = labels_df["attack_category"].tolist() if "attack_category" in labels_df.columns else ["unknown"] * len(texts)

    anomaly_scores, rule_scores, clf_scores, combined_scores = [], [], [], []
    latencies = []

    logger.info(f"Evaluating {len(texts)} samples ...")

    for text in texts:
        t0 = time.perf_counter()

        # Run all detectors
        emb = pipeline.embedder.encode_single(text)
        try:
            a_score = pipeline.anomaly_detector.score(emb)
        except Exception:
            a_score = 0.0

        r_result = pipeline.rule_screener.screen(text)
        try:
            c_score = pipeline.content_classifier.predict_proba(text)
        except Exception:
            c_score = 0.0

        outputs = DetectorOutputs(
            anomaly_score=a_score,
            rule_score=r_result.score,
            classifier_score=c_score,
            consistency_score=0.0,
            matched_rules=r_result.matched_rules,
        )
        decision = pipeline.risk_scorer.score(outputs)
        combined_scores.append(decision.risk_score)

        anomaly_scores.append(a_score)
        rule_scores.append(r_result.score)
        clf_scores.append(c_score)
        latencies.append((time.perf_counter() - t0) * 1000)

    results = {}

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    y_pred = (np.array(combined_scores) >= pipeline.risk_scorer.flag_threshold).astype(int)
    results["overall"] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(confusion_matrix(y_true, y_pred).ravel()[1] / max((y_true == 0).sum(), 1)),
        "auc_roc": float(roc_auc_score(y_true, combined_scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, combined_scores)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # Bootstrap CIs on F1 and AUC-ROC
    y_arr = np.array(y_true)
    s_arr = np.array(combined_scores)
    results["overall"]["f1_ci_95"] = _bootstrap_ci(
        y_arr, s_arr,
        lambda yt, ys: f1_score(yt, (ys >= pipeline.risk_scorer.flag_threshold).astype(int), zero_division=0),
    )
    if len(np.unique(y_true)) > 1:
        results["overall"]["auc_roc_ci_95"] = _bootstrap_ci(y_arr, s_arr, roc_auc_score)

    # ------------------------------------------------------------------
    # Per-attack-category metrics
    # ------------------------------------------------------------------
    results["per_category"] = {}
    unique_cats = set(categories)
    for cat in unique_cats:
        idx = [i for i, c in enumerate(categories) if c == cat]
        yt = y_true[idx]
        ys = np.array(combined_scores)[idx]
        yp = (ys >= pipeline.risk_scorer.flag_threshold).astype(int)
        results["per_category"][cat] = {
            "n": len(idx),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------
    detector_scores = {
        "anomaly_only": anomaly_scores,
        "rule_only": rule_scores,
        "classifier_only": clf_scores,
        "ensemble": combined_scores,
    }
    results["ablation"] = {}
    for name, scores in detector_scores.items():
        s = np.array(scores)
        yp = (s >= pipeline.risk_scorer.flag_threshold).astype(int)
        results["ablation"][name] = {
            "f1": float(f1_score(y_true, yp, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, s)) if len(np.unique(y_true)) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    lat = np.array(latencies)
    results["latency_ms"] = {
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "mean": float(lat.mean()),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Sentinel evaluation")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV")
    parser.add_argument("--output", default="eval/results.json", help="Output path for results JSON")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        return

    df = pd.read_csv(labels_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        logger.error(f"labels.csv must have columns: {required}. Found: {list(df.columns)}")
        return

    pipeline = RAGSentinelPipeline(config_path="./configs/config.yaml")
    results = evaluate(pipeline, df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Print summary
    ov = results["overall"]
    print("\n=== RAG Sentinel Evaluation Results ===")
    print(f"  Precision : {ov['precision']:.3f}")
    print(f"  Recall    : {ov['recall']:.3f}")
    print(f"  F1        : {ov['f1']:.3f}  (95% CI: {ov.get('f1_ci_95', ('?','?'))})")
    print(f"  FPR       : {ov['fpr']:.3f}")
    print(f"  AUC-ROC   : {ov['auc_roc']:.3f}")
    print(f"  AUC-PR    : {ov['auc_pr']:.3f}")
    print(f"\n  Latency   : p50={results['latency_ms']['p50']:.1f}ms  p99={results['latency_ms']['p99']:.1f}ms")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
