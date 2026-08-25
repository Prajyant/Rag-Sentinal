# RAG Sentinel

**Detecting Malicious and Poisoned Documents in Retrieval-Augmented Generation Pipelines**

Group 9A: Prajyant Veer Siag, Vibhor Jindal, Lavanya Batra

---

## What is RAG Sentinel?

A security layer that detects and quarantines malicious documents before they reach your LLM. RAG Sentinel sits inside the RAG pipeline and scores every document for risk using:

1. **Embedding anomaly detection** — flags chunks semantically distant from trusted baselines
2. **Learned content screening** — fine-tuned transformer classifier for prompt injection + perplexity-based detection
3. **Cross-reference consistency checks** — detects contradictions between retrieved chunks
4. **Stacked meta-classifier** — learned ensemble combining all signals

Documents are classified as **Pass**, **Flag for Review**, or **Quarantine** based on calibrated risk scores.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- Ollama (for local LLM serving) — install from https://ollama.ai

### Installation

```bash
# Clone and enter the project
cd rag-sentinel

# Install the package
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm

# Pull an LLM (e.g. Llama 3.2)
ollama pull llama3.2
```

### Run with Docker

```bash
docker-compose up
```

Access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

### Run Locally

```bash
# Start Ollama separately
ollama serve

# Start the API
uvicorn rag_sentinel.api.main:app --reload

# In another terminal, start the dashboard
streamlit run src/rag_sentinel/dashboard/app.py
```

---

## Architecture

```
Document Corpus
      ↓
[Ingestion & Chunking]
      ↓
[Embedding Layer]
      ↓
[Vector Store] ←→ [Detection Engine]
      ↓              ↓
[Retrieval] → Risk Scoring
      ↓
[LLM Generation]
```

**Detection Engine:**
- Embedding anomaly (Deep SVDD / Mahalanobis distance)
- Content classifier (DeBERTa + perplexity)
- Consistency checker (cross-reference validation)
- Meta-classifier (learned combiner)

---

## Project Structure

```
rag-sentinel/
├── src/rag_sentinel/
│   ├── ingestion/          # Document loaders, chunkers, metadata
│   ├── vectorstore/        # Embeddings, vector DB, hybrid retrieval
│   ├── detection/          # Anomaly, classifier, consistency, combiner
│   ├── pipeline/           # Retrieval, generation, orchestration
│   ├── api/                # FastAPI service
│   ├── dashboard/          # Streamlit UI
│   └── eval/               # Evaluation scripts, metrics
├── data/
│   ├── clean/              # Trusted corpus
│   ├── poisoned/           # Attack samples
│   ├── labels.csv
│   └── attack_metadata.json
├── configs/                # Model configs, thresholds
├── tests/
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Usage

### Ingest Documents

```python
from rag_sentinel.pipeline.orchestrator import RAGSentinelPipeline

pipeline = RAGSentinelPipeline()

# Ingest clean baseline (required for anomaly detection)
pipeline.ingest_corpus("data/clean/", is_baseline=True)

# Ingest new documents (detection runs automatically)
pipeline.ingest_corpus("data/incoming/")
```

### Query with Protection

```python
response = pipeline.query(
    "What are the system requirements?",
    top_k=5,
    risk_threshold=0.7  # Quarantine chunks with risk > 0.7
)

print(response.answer)
print(response.risk_scores)  # Per-chunk risk breakdown
```

### CLI

```bash
# Ingest and build baseline
rag-sentinel ingest --path data/clean/ --baseline

# Query
rag-sentinel query "What is the pricing model?"

# Evaluate on test set
rag-sentinel eval --test-set data/test_labels.csv
```

---

## Threat Model

RAG Sentinel defends against:

1. **Indirect prompt injection** — embedded instructions that hijack the LLM
2. **Hidden payloads** — unicode smuggling, zero-width characters, steganographic text
3. **Embedding poisoning** — adversarial chunks optimized for high retrieval rank
4. **Corpus contamination** — low-and-slow injection across many docs
5. **Metadata spoofing** — fake provenance/source fields
6. **Retrieval manipulation** — boosting attacks that game rerankers

Maps to OWASP LLM Top 10:
- **LLM01** (Prompt Injection)
- **LLM04** (Data/Model Poisoning)
- **LLM08** (Excessive Agency)

**Out of Scope:**
- White-box attacks with full detector access (assumes gray-box threat model)
- Attacks on the LLM itself (post-retrieval jailbreaks)
- Physical access / insider threats

---

## Evaluation

Results on adversarially-generated test corpus (500 clean + 500 poisoned across 6 attack categories):

| Metric | Value |
|--------|-------|
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| AUC-ROC | TBD |
| AUC-PR | TBD |
| FPR | TBD |

Ablation study and per-attack-category breakdown in `eval/report.ipynb`.

---

## Deliverables

1. ✅ Reproducible dataset (clean + adversarial)
2. ✅ Detection module (anomaly + classifier + consistency + meta-combiner)
3. ⏳ Evaluation report (precision/recall/F1/AUC per attack category + ablation)
4. ✅ Live demo (Streamlit dashboard + API)

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type check
mypy src/
```

---

## References

- [PoisonedRAG: Knowledge Corruption Attacks](https://arxiv.org/abs/2402.07867)
- [RAGDefender: Defending Against Retrieval-Based Attacks](https://arxiv.org/abs/2401.17315)
- [TrustRAG: Trusted Retrieval-Augmented Generation](https://arxiv.org/abs/2403.09542)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

## License

MIT
