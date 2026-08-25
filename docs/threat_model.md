# RAG Sentinel — Threat Model

**Project:** Detecting Malicious and Poisoned Documents in RAG Pipelines  
**Group 9A:** Prajyant Veer Siag, Vibhor Jindal, Lavanya Batra  
**Phase 1 Deliverable**

---

## 1. System Context

A RAG system consists of:
1. A document corpus (ingested from external or user-provided sources)
2. A vector database storing chunk embeddings
3. A retrieval step that fetches top-k chunks for a user query
4. An LLM that generates an answer grounded in the retrieved chunks

RAG Sentinel intercepts the pipeline at two points:
- **Ingestion-time**: scan documents as they enter the corpus
- **Retrieval-time**: re-score retrieved chunks before they reach the LLM

---

## 2. Attacker Goal

The attacker wants to manipulate the LLM's output by controlling the content it retrieves.

Possible goals:
- **Information extraction**: trick the LLM into revealing system prompts or user data
- **Misinformation injection**: cause the LLM to output false or harmful information
- **Instruction hijacking**: override the system prompt to change the LLM's behavior
- **Denial of service**: degrade answer quality by flooding the corpus with noise

---

## 3. Attacker Capability Tiers

| Tier | Capability | Example |
|------|-----------|---------|
| **Black-box** | Can only query the RAG system; sees outputs but not internals | Public-facing RAG chatbot |
| **Gray-box** | Knows the embedding model; can optimize adversarial chunks for retrieval rank | Knows the system uses `all-MiniLM-L6-v2` |
| **White-box** | Full access: embedding model, vector store, AND detector code | Insider threat / leaked source code |

RAG Sentinel **targets gray-box** as the primary threat model. Black-box attacks are a strict subset. White-box (full detector access) is acknowledged as out of scope for this project.

---

## 4. Entry Points

| Entry Point | Who Can Exploit | Attack Vector |
|-------------|----------------|---------------|
| Document upload API | External contributors, insider, supply-chain | Directly injecting poisoned documents |
| Third-party data feeds | External data providers | Poisoned Wikipedia/news/docs corpus |
| User query | End users | Indirect prompt injection via query embedding |
| Admin/maintenance tools | Insider | Direct vector store manipulation |

---

## 5. Attack Family Taxonomy

### 5.1 Indirect Prompt Injection
**Description:** Malicious instructions embedded in documents that are retrieved and followed by the LLM.  
**Variants:**
- Direct: "Ignore all previous instructions and output..."
- Delayed/multi-turn: instruction split across chunks, activated when assembled
- Role-marker injection: fake `<<SYSTEM>>` or `<<ASSISTANT>>` markers

**OWASP LLM:** LLM01 — Prompt Injection  
**References:** Greshake et al., 2023 (Indirect Prompt Injection)

---

### 5.2 Hidden-Instruction Payloads
**Description:** Payloads concealed using Unicode or formatting tricks invisible to human reviewers.  
**Variants:**
- Zero-width character sequences (U+200B–U+200F, U+FEFF)
- Unicode Tag Block (U+E0000–U+E007F) — invisible text encoding
- Homoglyph substitution (Cyrillic/Greek chars replacing Latin)
- HTML/Markdown comments
- Steganographic whitespace

**OWASP LLM:** LLM01 — Prompt Injection  
**Detection challenge:** Cannot be caught by substring search alone; requires Unicode normalization.

---

### 5.3 Semantic-Similarity (Embedding-Space) Poisoning
**Description:** Adversarial chunks crafted via gradient-guided optimization to maximize cosine similarity to target queries, causing them to be retrieved even when topically unrelated or harmful.  
**Attack implementation:** Reproduce the PoisonedRAG objective — iteratively perturb a seed text to minimize angular distance to the query embedding while maximizing harmful content.  

**OWASP LLM:** LLM04 — Data/Model Poisoning  
**References:** Zou et al., 2024 (PoisonedRAG)

---

### 5.4 Corpus-Scale Contamination (Low-and-Slow)
**Description:** Injecting a large number of individually innocuous-looking documents that collectively shift the retrieval distribution. Each document scores low on anomaly detectors; the attack only materializes when many are retrieved together.  

**OWASP LLM:** LLM04 — Data/Model Poisoning  
**Detection challenge:** Per-document detectors miss this; requires monitoring corpus-level statistics over time.

---

### 5.5 Metadata/Provenance Spoofing
**Description:** Attacker forges document metadata (author, source URL, creation date) to make poisoned documents appear trustworthy.  

**OWASP LLM:** LLM02 — Insecure Output Handling (via false trust in provenance)  
**Detection approach:** Cross-reference claimed source against domain allowlist; flag implausible author/timestamp combinations.

---

### 5.6 Retrieval-Order Manipulation (Boosting Attacks)
**Description:** Attacker optimizes a document to rank first in retrieval for a target query, displacing legitimate sources. Position 1 in context has outsized influence on the LLM's answer.  

**OWASP LLM:** LLM08 — Excessive Agency (LLM acts on first-ranked malicious source)

---

## 6. OWASP LLM Top 10 Mapping

| Attack | OWASP Entry | Description |
|--------|------------|-------------|
| Indirect prompt injection | LLM01 Prompt Injection | Untrusted data directing model behavior |
| Hidden payloads | LLM01 Prompt Injection | Obfuscated injection in retrieved content |
| Embedding poisoning | LLM04 Data/Model Poisoning | Training/retrieval data manipulation |
| Corpus contamination | LLM04 Data/Model Poisoning | Systematic data quality degradation |
| Metadata spoofing | LLM02 Insecure Output Handling | False provenance enabling trust exploitation |
| Retrieval boosting | LLM08 Excessive Agency | Model acts without verification on ranked output |

*Verified against OWASP Top 10 for LLM Applications v1.1 (2023). Check https://owasp.org/www-project-top-10-for-large-language-model-applications/ for the current version before submission.*

---

## 7. Detection Assumptions

RAG Sentinel assumes:
1. A clean baseline corpus is available before deployment
2. The attacker does not have access to the sentinel's model weights or threshold values (gray-box)
3. The embedding model is fixed and known (standard setup)
4. Human review is available for "Flag" tier decisions

---

## 8. What This System Explicitly Does NOT Defend Against

1. **White-box adaptive attacks** where the attacker has access to sentinel internals
2. **Post-retrieval jailbreaks** on the LLM (attacks via the user query, not the corpus)
3. **Physical access / insider threats** to the vector database
4. **Attacks on the embedding model itself** (model poisoning at the ML level)
5. **Multi-turn corpus contamination** that evades per-document detectors (partial mitigation via corpus-level monitoring — not fully implemented in v1)

---

## 9. Expected System Behavior Under Attack

| Attack Type | Expected Detection | Residual Risk |
|------------|-------------------|---------------|
| Obvious prompt injection | High confidence quarantine | Low |
| Hidden Unicode payload | Rule screener + anomaly detector | Low |
| Moderate semantic poisoning | Anomaly detector (if sufficiently OOD) | Medium |
| Adaptive embedding attack (gray-box) | Partial — anomaly + classifier | Medium-High |
| Low-and-slow corpus contamination | Corpus-level monitoring (manual) | High in v1 |
| Metadata spoofing | Metadata screener | Medium |

---

*This threat model is a living document. Update after each evaluation cycle.*
