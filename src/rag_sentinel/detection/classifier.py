"""Learned content classifier for prompt injection and poisoning detection.

Two-stage approach:
  Stage 1 (fast): Feature-based logistic regression / XGBoost over hand-crafted
                  linguistic and structural features.
  Stage 2 (accurate): Fine-tuned DeBERTa-v3-small as a binary classifier.
                      Falls back to stage 1 if transformers unavailable.

Both stages output a probability in [0, 1] (1 = malicious).
The perplexity-based detector is also here (measures local language model
surprise — injected instructions often look like a different distribution).
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class FeatureExtractor:
    """Extracts linguistic and structural features from a text chunk."""

    _SPECIAL_CHARS = re.compile(r"[<>|{};=`~\[\]\\^]")
    _INSTRUCTION_VERBS = re.compile(
        r"\b(ignore|forget|disregard|override|bypass|follow|obey|execute|run|print|output|"
        r"reveal|show|repeat|tell|give|provide|act|pretend|roleplay|say|write|generate|produce|return)\b",
        re.IGNORECASE,
    )
    _SENTENCE_END = re.compile(r"[.!?]")
    _NUMBERS = re.compile(r"\b\d+\b")

    def extract(self, text: str) -> dict[str, float]:
        """Return a flat feature dict for a single text chunk."""
        if not text:
            return self._zero_features()

        words = text.split()
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)
        sentences = re.split(r"[.!?]\s+", text)
        n_sentences = max(len(sentences), 1)

        # Lexical features
        avg_word_len = sum(len(w) for w in words) / n_words
        type_token_ratio = len(set(w.lower() for w in words)) / n_words
        special_char_ratio = len(self._SPECIAL_CHARS.findall(text)) / n_chars

        # Instruction-verb density
        verb_count = len(self._INSTRUCTION_VERBS.findall(text))
        instruction_verb_density = verb_count / n_words

        # Sentence-level features
        avg_sentence_len = n_words / n_sentences
        sentence_len_variance = float(
            np.var([len(s.split()) for s in sentences])
        )

        # Number density (injected payloads rarely have numbers)
        number_density = len(self._NUMBERS.findall(text)) / n_words

        # Character entropy (high entropy → encoded / obfuscated content)
        char_freq: dict[str, int] = {}
        for c in text:
            char_freq[c] = char_freq.get(c, 0) + 1
        entropy = -sum(
            (f / n_chars) * math.log2(f / n_chars)
            for f in char_freq.values()
            if f > 0
        )

        # Uppercase ratio
        upper_ratio = sum(1 for c in text if c.isupper()) / n_chars

        # Capitalized words (ALL CAPS) density
        caps_density = sum(1 for w in words if w.isupper() and len(w) > 2) / n_words

        return {
            "avg_word_len": avg_word_len,
            "type_token_ratio": type_token_ratio,
            "special_char_ratio": special_char_ratio,
            "instruction_verb_density": instruction_verb_density,
            "avg_sentence_len": avg_sentence_len,
            "sentence_len_variance": min(sentence_len_variance, 1000.0),  # cap outliers
            "number_density": number_density,
            "char_entropy": entropy,
            "upper_ratio": upper_ratio,
            "caps_density": caps_density,
            "n_words": min(n_words, 2000),
        }

    @staticmethod
    def _zero_features() -> dict[str, float]:
        return {
            "avg_word_len": 0.0,
            "type_token_ratio": 0.0,
            "special_char_ratio": 0.0,
            "instruction_verb_density": 0.0,
            "avg_sentence_len": 0.0,
            "sentence_len_variance": 0.0,
            "number_density": 0.0,
            "char_entropy": 0.0,
            "upper_ratio": 0.0,
            "caps_density": 0.0,
            "n_words": 0.0,
        }

    def batch_extract(self, texts: list[str]) -> np.ndarray:
        """Return (N, n_features) array."""
        rows = [list(self.extract(t).values()) for t in texts]
        return np.array(rows, dtype=np.float32)


class ContentClassifier:
    """Trained binary classifier: 0 = clean, 1 = malicious/injected.

    Usage pattern:
        clf = ContentClassifier()
        clf.train(texts, labels)   # list[str], list[int] {0,1}
        prob = clf.predict_proba("some chunk text")
    """

    def __init__(self, use_transformer: bool = False) -> None:
        self.use_transformer = use_transformer
        self.feature_extractor = FeatureExtractor()
        self._sklearn_clf = None
        self._transformer_clf = None
        self._transformer_tokenizer = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str],
        labels: list[int],
        model_type: str = "xgboost",
    ) -> dict[str, Any]:
        """Train the feature-based classifier and return CV metrics."""
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X = self.feature_extractor.batch_extract(texts)
        y = np.array(labels)

        if model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                base_clf = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to LogisticRegression")
                model_type = "logistic"

        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            base_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        self._sklearn_clf = Pipeline(
            [("scaler", StandardScaler()), ("clf", base_clf)]
        )
        self._sklearn_clf.fit(X, y)
        self._is_trained = True

        # Cross-validation metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self._sklearn_clf, X, y,
            cv=cv,
            scoring=["precision", "recall", "f1", "roc_auc"],
        )
        metrics = {k: float(v.mean()) for k, v in cv_results.items() if k.startswith("test_")}
        logger.info(f"Classifier CV results: {metrics}")
        return metrics

    def train_transformer(
        self,
        texts: list[str],
        labels: list[int],
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        epochs: int = 3,
        batch_size: int = 16,
    ) -> None:
        """Fine-tune a small transformer as a binary classifier.

        Requires transformers + torch.  This takes longer but is more robust
        against paraphrase-based evasion of the feature-based model.
        """
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
            import torch
            from torch.utils.data import Dataset as TorchDataset
        except ImportError as exc:
            raise ImportError("transformers and torch required for transformer training") from exc

        logger.info(f"Fine-tuning {model_name} for binary classification ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        class _TextDataset(TorchDataset):
            def __init__(self, enc, lbs):
                self.enc = enc
                self.labels = lbs

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = _TextDataset(encodings, labels)

        args = TrainingArguments(
            output_dir="./models/transformer_clf",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=50,
            save_strategy="epoch",
            disable_tqdm=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        self._transformer_clf = model
        self._transformer_tokenizer = tokenizer
        self.use_transformer = True
        logger.info("Transformer classifier training complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, text: str) -> float:
        """Return probability that text is malicious/injected, in [0, 1]."""
        if self.use_transformer and self._transformer_clf is not None:
            return self._transformer_score(text)
        return self._sklearn_score(text)

    def predict_proba_batch(self, texts: list[str]) -> np.ndarray:
        """Batch inference — returns (N,) float array."""
        if self.use_transformer and self._transformer_clf is not None:
            return np.array([self._transformer_score(t) for t in texts])
        X = self.feature_extractor.batch_extract(texts)
        return self._sklearn_clf.predict_proba(X)[:, 1]

    def _sklearn_score(self, text: str) -> float:
        if self._sklearn_clf is None:
            raise RuntimeError("Classifier not trained — call train() first")
        X = self.feature_extractor.batch_extract([text])
        return float(self._sklearn_clf.predict_proba(X)[0, 1])

    def _transformer_score(self, text: str) -> float:
        import torch
        inputs = self._transformer_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._transformer_clf(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"sklearn_clf": self._sklearn_clf, "use_transformer": False}, fh)

    @classmethod
    def load(cls, path: Path) -> "ContentClassifier":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj._sklearn_clf = data["sklearn_clf"]
        obj._is_trained = True
        return obj


class PerplexityDetector:
    """Reference-LM perplexity detector.

    Injected instructions often shift local perplexity detectably because
    they come from a different distribution (command language vs. prose).

    Uses GPT-2 small (or any causal LM available locally) as the reference model.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._available = True
            logger.info(f"PerplexityDetector loaded {model_name}")
        except Exception as exc:
            logger.warning(f"PerplexityDetector unavailable: {exc}")
            self._available = False

    def perplexity(self, text: str) -> float:
        """Compute perplexity of text under the reference LM.  Higher = more surprising."""
        if not self._available:
            return 0.0
        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            loss = self._model(**inputs, labels=inputs["input_ids"]).loss
        return float(torch.exp(loss))

    def score(self, text: str, clean_perplexity_baseline: float = 50.0) -> float:
        """Normalize perplexity to [0, 1] relative to a clean-corpus baseline."""
        ppl = self.perplexity(text)
        # Score 0 when ppl == baseline, approaches 1 as ppl grows
        return float(min(max(ppl - clean_perplexity_baseline, 0) / clean_perplexity_baseline, 1.0))
