"""Streamlit dashboard for RAG Sentinel.

Shows:
  - Live query interface with risk score visualization
  - Per-chunk detector breakdown
  - Running decision history (pass / flag / quarantine)
  - Real-time ROC curve (updates as decisions are logged)

Run with:
  streamlit run src/rag_sentinel/dashboard/app.py
"""

from __future__ import annotations

import time
import json
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

API_BASE = "http://localhost:8000"

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Sentinel",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ RAG Sentinel — Live Detection Dashboard")
st.caption("Security layer for Retrieval-Augmented Generation pipelines")

# ------------------------------------------------------------------
# Sidebar: ingest controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Corpus Management")

    ingest_path = st.text_input("Document path", value="./data/clean/")
    is_baseline = st.checkbox("Mark as trusted baseline", value=True)
    if st.button("Ingest Documents"):
        try:
            r = requests.post(
                f"{API_BASE}/ingest",
                json={"path": ingest_path, "is_baseline": is_baseline},
                timeout=120,
            )
            if r.ok:
                data = r.json()
                st.success(f"Ingested {data['chunks_ingested']} chunks")
            else:
                st.error(f"Error: {r.text}")
        except Exception as exc:
            st.error(f"Could not reach API: {exc}")

    st.divider()

    st.header("Stats")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
        st.metric("Total chunks", stats.get("total_chunks", "—"))
        st.metric("Baseline samples", stats.get("baseline_samples", "—"))
        st.caption(f"Model: {stats.get('embedding_model', '—')}")
    except Exception:
        st.warning("API not reachable")

# ------------------------------------------------------------------
# Main panel: query + results
# ------------------------------------------------------------------
col_query, col_results = st.columns([1, 1])

with col_query:
    st.subheader("Ask a Question")
    question = st.text_area("Query", height=100, placeholder="e.g. What are the system requirements?")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    risk_thresh = st.slider("Quarantine threshold", 0.0, 1.0, 0.70, 0.05)

    if st.button("Submit Query", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running Sentinel pipeline ..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "max_risk_score": risk_thresh,
                        },
                        timeout=60,
                    )
                    if r.ok:
                        st.session_state["last_response"] = r.json()
                    else:
                        st.error(f"API error: {r.text}")
                except Exception as exc:
                    st.error(f"Could not reach API: {exc}")

with col_results:
    st.subheader("Response")
    resp = st.session_state.get("last_response")
    if resp:
        st.write(resp.get("answer", ""))
        st.caption(
            f"⏱ {resp['latency_ms']:.0f} ms | "
            f"✅ {resp['passed_chunks']} passed  "
            f"⚠️ {resp['flagged_chunks']} flagged  "
            f"🚫 {resp['quarantined_chunks']} quarantined"
        )

# ------------------------------------------------------------------
# Risk score table
# ------------------------------------------------------------------
if resp and resp.get("risk_decisions"):
    st.subheader("Chunk Risk Breakdown")
    rows = []
    for d in resp["risk_decisions"]:
        rows.append({
            "Chunk ID": d["chunk_id"][-40:],
            "Risk Score": round(d["risk_score"], 3),
            "Decision": d["decision"].upper(),
            "Anomaly": round(d["anomaly_score"], 3),
            "Rule": round(d["rule_score"], 3),
            "Classifier": round(d["classifier_score"], 3),
            "Consistency": round(d["consistency_score"], 3),
            "Rules Hit": ", ".join(d.get("matched_rules", [])[:3]),
        })
    df = pd.DataFrame(rows)

    def _color_decision(val):
        colors = {"PASS": "background-color: #d4edda", "FLAG": "background-color: #fff3cd", "QUARANTINE": "background-color: #f8d7da"}
        return colors.get(val, "")

    styled = df.style.applymap(_color_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True)

# ------------------------------------------------------------------
# Decision history
# ------------------------------------------------------------------
st.divider()
st.subheader("Recent Decision History")

try:
    hist = requests.get(f"{API_BASE}/decisions?limit=100", timeout=5).json()
    decisions = hist.get("decisions", [])
    if decisions:
        hdf = pd.DataFrame(decisions)
        counts = hdf["decision"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Passed", counts.get("pass", 0))
        c2.metric("⚠️ Flagged", counts.get("flag", 0))
        c3.metric("🚫 Quarantined", counts.get("quarantine", 0))

        # Risk score distribution
        fig, ax = plt.subplots(figsize=(6, 2.5))
        hdf["risk_score"].hist(bins=20, ax=ax, color="#4a90e2", edgecolor="white")
        ax.axvline(0.3, color="orange", linestyle="--", label="Flag threshold")
        ax.axvline(0.7, color="red", linestyle="--", label="Quarantine threshold")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title("Risk Score Distribution")
        st.pyplot(fig)
    else:
        st.info("No decisions logged yet — submit a query above.")
except Exception:
    st.info("API not reachable — start the API first.")

st.caption("RAG Sentinel — Group 9A | Auto-refreshes on new queries")
