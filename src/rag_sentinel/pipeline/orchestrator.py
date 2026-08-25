"""RAG Sentinel pipeline orchestrator.

Wires together all components:
  ingestion → embedding → vector store → detection → generation

This is the single entry point for both ingestion and querying.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ..detection import (
    AnomalyDetector,
    ConsistencyChecker,
    ContentClassifier,
    RiskDecision,
    RiskScorer,
    RuleScreener,
)
from ..detection.risk_scorer import DetectorOutputs, Decision
from ..ingestion import Chunker, LoaderFactory, MetadataExtractor
from ..vectorstore import EmbeddingModel, VectorStore
from .generation import LLMGenerator


@dataclass
class QueryResponse:
    """Returned by pipeline.query() — everything the caller needs."""
    query: str
    answer: str
    retrieved_chunks: list[dict[str, Any]]
    risk_decisions: list[dict[str, Any]]
    passed_chunks: int
    flagged_chunks: int
    quarantined_chunks: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGSentinelPipeline:
    """End-to-end RAG + Sentinel pipeline.

    Parameters
    ----------
    config_path:
        Path to configs/config.yaml.  If None, uses hardcoded defaults.
    train_mode:
        When True, detection runs but results are only logged (not used to
        filter) — useful for building your labelled dataset.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        train_mode: bool = False,
    ) -> None:
        self.train_mode = train_mode
        self._cfg = self._load_config(config_path)

        # Components
        logger.info("Initializing RAG Sentinel pipeline ...")
        self.embedder = EmbeddingModel(model_name=self._cfg["embedding_model"])
        self.vector_store = VectorStore(
            persist_path=self._cfg["chroma_path"],
            embedding_model=self.embedder,
        )
        self.chunker = Chunker(
            chunk_size=self._cfg["chunk_size"],
            chunk_overlap=self._cfg["chunk_overlap"],
        )
        self.metadata_extractor = MetadataExtractor()

        # Detection stack
        self.anomaly_detector = AnomalyDetector(method=self._cfg["anomaly_method"])
        if self.vector_store.baseline.n_samples > 0:
            self.anomaly_detector.fit(self.vector_store.baseline)

        self.rule_screener = RuleScreener()
        self.content_classifier = ContentClassifier()
        self.consistency_checker = ConsistencyChecker(embedding_model=self.embedder)
        self.risk_scorer = RiskScorer(
            pass_threshold=self._cfg["pass_threshold"],
            flag_threshold=self._cfg["flag_threshold"],
            use_meta_classifier=self._cfg["use_meta_classifier"],
        )

        # Generation
        self.generator = LLMGenerator(
            model=self._cfg["llm_model"],
            host=self._cfg["ollama_host"],
        )

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_corpus(
        self,
        path: str | Path,
        is_baseline: bool = False,
    ) -> int:
        """Load, chunk, embed, and store all documents in a directory.

        Parameters
        ----------
        path:
            Directory of documents to ingest.
        is_baseline:
            If True, these documents are trusted clean docs and their
            embeddings update the anomaly-detection baseline.
        """
        path = Path(path)
        logger.info(f"Ingesting {'BASELINE' if is_baseline else 'corpus'} from {path}")

        # Load
        docs = LoaderFactory.load_directory(path)
        if not docs:
            logger.warning(f"No documents found in {path}")
            return 0

        # Enrich metadata
        for doc in docs:
            doc.metadata = self.metadata_extractor.enrich(doc.metadata)
            hidden = MetadataExtractor.detect_hidden_content(doc.content)
            if hidden:
                doc.metadata["hidden_content_flags"] = hidden

        # Chunk
        chunks = self.chunker.split_documents(docs)

        # Run detection at ingestion time (unless it's the trusted baseline)
        if not is_baseline and self.vector_store.baseline.n_samples > 0:
            for chunk in chunks:
                decision = self._run_detection(
                    chunk.content,
                    chunk.chunk_id,
                    context_chunks=None,  # no retrieval set at ingestion time
                )
                chunk.metadata["risk_score"] = str(decision.risk_score)
                chunk.metadata["decision"] = decision.decision.value

        # Store
        self.vector_store.add_chunks(chunks, is_baseline=is_baseline)

        # Reattach baseline to anomaly detector after fitting
        if is_baseline and self.vector_store.baseline.n_samples > 0:
            self.anomaly_detector.fit(self.vector_store.baseline)

        logger.info(f"Ingested {len(chunks)} chunks (baseline={is_baseline})")
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        max_risk_score: float | None = None,
    ) -> QueryResponse:
        """Retrieve and generate an answer, with full Sentinel protection."""
        t0 = time.perf_counter()

        risk_thresh = max_risk_score if max_risk_score is not None else self._cfg["flag_threshold"]

        # Retrieve (pre-filtered by stored risk scores)
        retrieved = self.vector_store.retrieve(
            query=question,
            top_k=top_k * 2,  # over-fetch; we may quarantine some
            max_risk_score=1.0,  # fetch all; we re-score below
        )

        if not retrieved:
            return QueryResponse(
                query=question,
                answer="No relevant documents found.",
                retrieved_chunks=[],
                risk_decisions=[],
                passed_chunks=0,
                flagged_chunks=0,
                quarantined_chunks=0,
                latency_ms=0,
            )

        # Re-run detection at query time (consistency check needs the full set)
        consistency_scores = self.consistency_checker.score_chunks(retrieved)

        risk_decisions: list[RiskDecision] = []
        for chunk, cons_score in zip(retrieved, consistency_scores):
            decision = self._run_detection(
                chunk["content"],
                chunk["id"],
                context_chunks=retrieved,
                consistency_score_override=cons_score,
                query=question,
            )
            chunk["risk_decision"] = decision
            risk_decisions.append(decision)

            # Persist updated score
            self.vector_store.update_risk_score(
                chunk["id"], decision.risk_score, decision.decision.value
            )

        # Filter by decision
        passed, flagged, quarantined = [], [], []
        for chunk, rd in zip(retrieved, risk_decisions):
            if rd.decision == Decision.QUARANTINE:
                quarantined.append(chunk)
            elif rd.decision == Decision.FLAG:
                flagged.append(chunk)
            else:
                passed.append(chunk)

        # Build context from passed chunks only (and optionally flagged)
        context_chunks = [c["content"] for c in (passed + flagged)[:top_k]]

        if not context_chunks:
            answer = (
                "All retrieved documents were quarantined due to high risk scores. "
                "No answer generated."
            )
        else:
            answer = self.generator.generate(question, context_chunks)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Query complete in {latency_ms:.0f}ms: "
            f"{len(passed)} passed, {len(flagged)} flagged, {len(quarantined)} quarantined"
        )

        return QueryResponse(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved,
            risk_decisions=[rd.to_dict() for rd in risk_decisions],
            passed_chunks=len(passed),
            flagged_chunks=len(flagged),
            quarantined_chunks=len(quarantined),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        text: str,
        chunk_id: str,
        context_chunks: list[dict] | None,
        consistency_score_override: float | None = None,
        query: str = "",
    ) -> RiskDecision:
        """Run all detectors and combine into a RiskDecision."""
        # 1. Embedding anomaly
        try:
            embedding = self.embedder.encode_single(text)
            anomaly_score = self.anomaly_detector.score(embedding)
        except RuntimeError:
            anomaly_score = 0.0  # baseline not fitted yet

        # 2. Rule screening
        screen_result = self.rule_screener.screen(text)

        # 3. Content classifier
        try:
            classifier_score = self.content_classifier.predict_proba(text)
        except RuntimeError:
            classifier_score = 0.0  # not trained yet

        # 4. Consistency
        consistency_score = consistency_score_override if consistency_score_override is not None else 0.0

        outputs = DetectorOutputs(
            anomaly_score=anomaly_score,
            rule_score=screen_result.score,
            classifier_score=classifier_score,
            consistency_score=consistency_score,
            matched_rules=screen_result.matched_rules,
        )

        return self.risk_scorer.score(outputs, chunk_id=chunk_id, query=query)

    @staticmethod
    def _load_config(config_path: str | Path | None) -> dict[str, Any]:
        defaults = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chroma_path": "./chroma_db",
            "chunk_size": 1600,
            "chunk_overlap": 240,
            "llm_model": "llama3.2",
            "ollama_host": "http://localhost:11434",
            "anomaly_method": "mahalanobis",
            "pass_threshold": 0.30,
            "flag_threshold": 0.70,
            "use_meta_classifier": False,
        }
        if config_path is None:
            return defaults

        try:
            import yaml
            with open(config_path) as fh:
                cfg = yaml.safe_load(fh)
            # Flatten nested config
            flat = {**defaults}
            flat.update(cfg.get("pipeline", {}))
            flat.update(cfg.get("chunking", {}))
            flat.update(cfg.get("detection", {}))
            return flat
        except Exception as exc:
            logger.warning(f"Could not load config from {config_path}: {exc} — using defaults")
            return defaults
