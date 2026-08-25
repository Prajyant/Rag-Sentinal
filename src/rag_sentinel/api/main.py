"""FastAPI service for RAG Sentinel.

Endpoints:
  POST /ingest          — ingest a directory of documents
  POST /query           — query with Sentinel protection
  GET  /stats           — pipeline statistics
  GET  /decisions       — recent risk decisions
  GET  /health          — health check
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from ..pipeline import RAGSentinelPipeline


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class IngestRequest(BaseModel):
    path: str
    is_baseline: bool = False


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    max_risk_score: float = 0.70


class IngestResponse(BaseModel):
    chunks_ingested: int
    path: str
    is_baseline: bool


class QueryApiResponse(BaseModel):
    query: str
    answer: str
    passed_chunks: int
    flagged_chunks: int
    quarantined_chunks: int
    latency_ms: float
    risk_decisions: list[dict[str, Any]]


# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

_pipeline: RAGSentinelPipeline | None = None
_decision_log: list[dict] = []   # in-memory ring buffer for the dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Starting RAG Sentinel API ...")
    _pipeline = RAGSentinelPipeline(config_path="./configs/config.yaml")
    yield
    logger.info("Shutting down ...")


app = FastAPI(
    title="RAG Sentinel",
    description="Security layer for RAG pipelines — detects malicious and poisoned documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    n = _pipeline.ingest_corpus(req.path, is_baseline=req.is_baseline)
    return IngestResponse(chunks_ingested=n, path=req.path, is_baseline=req.is_baseline)


@app.post("/query", response_model=QueryApiResponse)
def query(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    response = _pipeline.query(
        question=req.question,
        top_k=req.top_k,
        max_risk_score=req.max_risk_score,
    )

    # Log to ring buffer (keep last 200)
    _decision_log.extend(response.risk_decisions)
    if len(_decision_log) > 200:
        del _decision_log[: len(_decision_log) - 200]

    return QueryApiResponse(
        query=response.query,
        answer=response.answer,
        passed_chunks=response.passed_chunks,
        flagged_chunks=response.flagged_chunks,
        quarantined_chunks=response.quarantined_chunks,
        latency_ms=response.latency_ms,
        risk_decisions=response.risk_decisions,
    )


@app.get("/decisions")
def get_decisions(limit: int = 50):
    """Return the most recent risk decisions — consumed by the dashboard."""
    return {"decisions": _decision_log[-limit:]}


@app.get("/stats")
def stats():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "total_chunks": _pipeline.vector_store.count,
        "baseline_samples": _pipeline.vector_store.baseline.n_samples,
        "embedding_model": _pipeline.embedder.model_name,
        "embedding_dim": _pipeline.embedder.dim,
    }
