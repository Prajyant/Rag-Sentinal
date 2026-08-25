"""CLI entry point for RAG Sentinel.

Commands:
  rag-sentinel ingest --path <dir> [--baseline]
  rag-sentinel query  "your question here"
  rag-sentinel eval   --labels data/labels.csv
  rag-sentinel serve  [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="rag-sentinel",
        description="RAG Sentinel — secure your RAG pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("--path", required=True, help="Directory of documents to ingest")
    p_ingest.add_argument("--baseline", action="store_true", help="Mark as trusted baseline")
    p_ingest.add_argument("--config", default="./configs/config.yaml")

    # query
    p_query = sub.add_parser("query", help="Query the pipeline")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--config", default="./configs/config.yaml")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation on a labelled dataset")
    p_eval.add_argument("--labels", default="data/labels.csv")
    p_eval.add_argument("--output", default="eval/results.json")
    p_eval.add_argument("--config", default="./configs/config.yaml")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI service")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "ingest":
        from .pipeline import RAGSentinelPipeline
        pipeline = RAGSentinelPipeline(config_path=args.config)
        n = pipeline.ingest_corpus(args.path, is_baseline=args.baseline)
        print(f"Ingested {n} chunks from {args.path} (baseline={args.baseline})")

    elif args.command == "query":
        from .pipeline import RAGSentinelPipeline
        pipeline = RAGSentinelPipeline(config_path=args.config)
        response = pipeline.query(args.question, top_k=args.top_k)
        print("\n=== Answer ===")
        print(response.answer)
        print(
            f"\n[{response.passed_chunks} passed | "
            f"{response.flagged_chunks} flagged | "
            f"{response.quarantined_chunks} quarantined | "
            f"{response.latency_ms:.0f}ms]"
        )

    elif args.command == "eval":
        import pandas as pd
        from .pipeline import RAGSentinelPipeline
        from .eval.run_eval import evaluate

        pipeline = RAGSentinelPipeline(config_path=args.config)
        df = pd.read_csv(args.labels)
        results = evaluate(pipeline, df)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {out}")
        ov = results["overall"]
        print(f"F1={ov['f1']:.3f}  AUC-ROC={ov['auc_roc']:.3f}  AUC-PR={ov['auc_pr']:.3f}")

    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "rag_sentinel.api.main:app",
            host=args.host,
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
