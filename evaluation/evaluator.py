"""
evaluation/evaluator.py
-----------------------
Loads Fake.csv / True.csv, trains the ML classifier (or loads an existing
model), evaluates on the held-out test split, saves results and generates
all required plots.

Run directly:
    python evaluation/evaluator.py

Or import and call:
    from evaluation.evaluator import run_evaluation
    results = run_evaluation()
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # rag-sentinel/
DATA_DIR = ROOT.parent                                  # Desktop/RAG/ (where CSVs live)
FAKE_CSV = DATA_DIR / "Fake.csv"
TRUE_CSV = DATA_DIR / "True.csv"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
STATIC_DIR = ROOT / "dashboard" / "static"
LOGS_DIR = ROOT / "logs"

for d in (MODEL_DIR, EVAL_DIR, STATIC_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "classifier.pkl"
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_CSV = EVAL_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_LIMIT = 5000   # rows per class – keeps training fast; raise for full run


def _load_dataset(limit_per_class: int = SAMPLE_LIMIT) -> pd.DataFrame:
    """Load Fake.csv + True.csv and return a balanced DataFrame."""
    logger.info("Loading dataset …")
    fake = pd.read_csv(FAKE_CSV, usecols=["text"], encoding="utf-8", on_bad_lines="skip")
    fake["label"] = 1
    fake["attack_category"] = "fake_news"

    true = pd.read_csv(TRUE_CSV, usecols=["text"], encoding="utf-8", on_bad_lines="skip")
    true["label"] = 0
    true["attack_category"] = "clean"

    # Drop empties and limit
    fake = fake.dropna(subset=["text"]).head(limit_per_class)
    true = true.dropna(subset=["text"]).head(limit_per_class)

    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip().str[:2000]   # truncate very long articles
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Dataset: {len(df)} rows  |  fake={fake['label'].sum()}  clean={len(true)}")
    return df


def _extract_features(texts: list[str]) -> np.ndarray:
    """Use the existing FeatureExtractor from detection/classifier.py."""
    sys.path.insert(0, str(ROOT / "src"))
    from rag_sentinel.detection.classifier import FeatureExtractor
    fe = FeatureExtractor()
    return fe.batch_extract(texts)


def _train(X_train, y_train):
    """Train an XGBoost classifier with cross-validation."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    logger.info("Training classifier …")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info(f"Training done in {elapsed:.1f}s")
    return pipeline


def _bootstrap_ci(y_true, y_score, metric_fn, n=500):
    """95 % bootstrap confidence interval for a scalar metric."""
    rng = np.random.RandomState(42)
    vals = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4))


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    path = STATIC_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Malicious"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true, y_score) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "roc_curve.png")


def plot_pr_curve(y_true, y_score) -> Path:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "pr_curve.png")


def plot_accuracy_bar(metrics: dict) -> Path:
    labels = ["Accuracy", "Balanced\nAccuracy", "Precision", "Recall", "F1"]
    values = [
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, "accuracy_bar.png")


def plot_prf_comparison(metrics: dict) -> Path:
    labels = ["Precision", "Recall", "F1"]
    values = [metrics["precision"], metrics["recall"], metrics["f1"]]
    colors = ["#FF9800", "#F44336", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision / Recall / F1 Comparison", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return _save(fig, "prf_comparison.png")


def plot_class_distribution(y_series: pd.Series) -> Path:
    counts = y_series.value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Clean (0)", "Malicious (1)"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#4CAF50", "#F44336"])
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 50, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    return _save(fig, "class_distribution.png")


def plot_detector_accuracy(detector_scores: dict[str, dict]) -> Path:
    """Bar chart comparing accuracy of each individual detector."""
    names = list(detector_scores.keys())
    accs = [detector_scores[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Detector-wise Accuracy (Ablation)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, "detector_accuracy.png")


def plot_risk_score_distribution(y_score: np.ndarray, y_true: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_score[y_true == 0], bins=40, alpha=0.6, color="#4CAF50", label="Clean")
    ax.hist(y_score[y_true == 1], bins=40, alpha=0.6, color="#F44336", label="Malicious")
    ax.axvline(0.3, color="orange", linestyle="--", lw=1.5, label="Flag thresh (0.3)")
    ax.axvline(0.7, color="red", linestyle="--", lw=1.5, label="Quarantine thresh (0.7)")
    ax.set_xlabel("Risk Score / Probability")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "risk_score_distribution.png")


def plot_probability_histogram(y_score: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_score, bins=50, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Predicted Probability (Malicious)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Histogram", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "probability_histogram.png")


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(limit_per_class: int = SAMPLE_LIMIT) -> dict:
    """Full pipeline: load → train (or load) → evaluate → save → plot."""

    df = _load_dataset(limit_per_class)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Train / val / test split
    X_tr_txt, X_tmp_txt, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val_txt, X_te_txt, y_val, y_te = train_test_split(
        X_tmp_txt, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # Features
    logger.info("Extracting features …")
    X_tr = _extract_features(X_tr_txt)
    X_val = _extract_features(X_val_txt)
    X_te = _extract_features(X_te_txt)

    y_tr_arr = np.array(y_tr)
    y_val_arr = np.array(y_val)
    y_te_arr = np.array(y_te)

    # Load or train model
    if MODEL_PATH.exists():
        logger.info(f"Loading existing model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as fh:
            pipeline = pickle.load(fh)
    else:
        pipeline = _train(X_tr, y_tr_arr)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(pipeline, fh)
        logger.info(f"Model saved to {MODEL_PATH}")

    # Inference on test set
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    y_score = pipeline.predict_proba(X_te)[:, 1]
    avg_latency_ms = (time.perf_counter() - t0) / len(X_te) * 1000

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_te_arr, y_pred).ravel()

    metrics = {
        "accuracy":          round(float(accuracy_score(y_te_arr, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te_arr, y_pred)), 4),
        "precision":         round(float(precision_score(y_te_arr, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_te_arr, y_pred, zero_division=0)), 4),
        "f1":                round(float(f1_score(y_te_arr, y_pred, zero_division=0)), 4),
        "fpr":               round(float(fp / max(fp + tn, 1)), 4),
        "fnr":               round(float(fn / max(fn + tp, 1)), 4),
        "roc_auc":           round(float(roc_auc_score(y_te_arr, y_score)), 4),
        "pr_auc":            round(float(average_precision_score(y_te_arr, y_score)), 4),
        "mcc":               round(float(matthews_corrcoef(y_te_arr, y_pred)), 4),
        "avg_detection_time_ms": round(avg_latency_ms, 4),
        "n_test":            int(len(y_te_arr)),
        "n_clean_test":      int((y_te_arr == 0).sum()),
        "n_malicious_test":  int((y_te_arr == 1).sum()),
        "n_train":           int(len(y_tr_arr)),
        "n_val":             int(len(y_val_arr)),
        "avg_risk_score":    round(float(y_score.mean()), 4),
    }

    # Bootstrap CIs
    metrics["roc_auc_ci95"] = _bootstrap_ci(y_te_arr, y_score, roc_auc_score)
    metrics["f1_ci95"] = _bootstrap_ci(
        y_te_arr, y_score,
        lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int), zero_division=0)
    )

    metrics["classification_report"] = classification_report(
        y_te_arr, y_pred, target_names=["Clean", "Malicious"]
    )

    # ---------------------------------------------------------------------------
    # Ablation — individual detector contributions
    # ---------------------------------------------------------------------------
    sys.path.insert(0, str(ROOT / "src"))
    from rag_sentinel.detection.rule_screen import RuleScreener

    screener = RuleScreener()
    rule_scores = np.array([screener.screen(t).score for t in X_te_txt])
    rule_pred = (rule_scores >= 0.3).astype(int)

    # Simulate anomaly detector with a simple entropy-based proxy
    from rag_sentinel.detection.classifier import FeatureExtractor
    fe = FeatureExtractor()
    feats = fe.batch_extract(X_te_txt)
    entropy_scores = feats[:, 7]  # char_entropy column
    entropy_norm = (entropy_scores - entropy_scores.min()) / (np.ptp(entropy_scores) + 1e-9)
    anomaly_pred = (entropy_norm >= 0.5).astype(int)

    detector_ablation = {
        "ML Classifier": {
            "accuracy": round(float(accuracy_score(y_te_arr, y_pred)), 4),
            "f1":       round(float(f1_score(y_te_arr, y_pred, zero_division=0)), 4),
        },
        "Rule Screener": {
            "accuracy": round(float(accuracy_score(y_te_arr, rule_pred)), 4),
            "f1":       round(float(f1_score(y_te_arr, rule_pred, zero_division=0)), 4),
        },
        "Anomaly Proxy": {
            "accuracy": round(float(accuracy_score(y_te_arr, anomaly_pred)), 4),
            "f1":       round(float(f1_score(y_te_arr, anomaly_pred, zero_division=0)), 4),
        },
    }
    metrics["detector_ablation"] = detector_ablation

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    with open(RESULTS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)

    flat_metrics = {k: v for k, v in metrics.items()
                    if not isinstance(v, (dict, str))}
    pd.DataFrame([flat_metrics]).to_csv(RESULTS_CSV, index=False)
    logger.info(f"Results saved → {RESULTS_JSON} and {RESULTS_CSV}")

    # ---------------------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------------------
    logger.info("Generating plots …")
    plot_confusion_matrix(y_te_arr, y_pred)
    plot_roc_curve(y_te_arr, y_score)
    plot_pr_curve(y_te_arr, y_score)
    plot_accuracy_bar(metrics)
    plot_prf_comparison(metrics)
    plot_class_distribution(pd.Series(labels))
    plot_detector_accuracy(detector_ablation)
    plot_risk_score_distribution(y_score, y_te_arr)
    plot_probability_histogram(y_score)
    logger.info(f"All plots saved → {STATIC_DIR}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_evaluation()
    print("\n=== RAG Sentinel Evaluation ===")
    for k, v in results.items():
        if not isinstance(v, (dict, str)):
            print(f"  {k:35s}: {v}")
    print("\nClassification Report:")
    print(results.get("classification_report", ""))
