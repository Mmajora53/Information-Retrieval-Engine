<div align="center">

# Information Retrieval Engine

### Stack Exchange Document Ranking — Kaggle Competition

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-5.2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://www.sbert.net)
[![BM25+](https://img.shields.io/badge/BM25%2B-rank--bm25-4B8BBE?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)
[![Jupyter](https://img.shields.io/badge/Jupyter-4.5-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Team%20Rocket-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/t/f383bc6c1f194226bb43d21ab3d65418)

<br/>

**Best Score `0.49541`** &nbsp;·&nbsp; **Classifier Macro-F1 `0.9852`** &nbsp;·&nbsp; **216,041 documents** &nbsp;·&nbsp; **141 queries**

<br/>

> **🌐 Language:** &nbsp;🇬🇧 English &nbsp;|&nbsp; [🇫🇷 Lire en Français](README.md.fr)

</div>

---

## Table of Contents

- [Information Retrieval Engine](#information-retrieval-engine)
    - [Stack Exchange Document Ranking — Kaggle Competition](#stack-exchange-document-ranking--kaggle-competition)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Pipeline](#pipeline)
    - [Phase 1 — Hybrid Retrieval](#phase-1--hybrid-retrieval)
    - [Phase 2 — Classification \& Reranking](#phase-2--classification--reranking)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Data Setup](#data-setup)
  - [Usage](#usage)
    - [Phase 1 — Hybrid Retrieval](#phase-1--hybrid-retrieval-1)
    - [Phase 2 — Classification \& Reranking](#phase-2--classification--reranking-1)
    - [Notebooks Overview](#notebooks-overview)
  - [Configuration](#configuration)
  - [Results](#results)
    - [Kaggle Leaderboard](#kaggle-leaderboard)
    - [Classifier Ablation (notebook 07)](#classifier-ablation-notebook-07)
    - [Per-Category Performance (best model)](#per-category-performance-best-model)
  - [Authors](#authors)

---

## Overview

This project is a **two-phase information retrieval system** built for a university Kaggle competition on the Stack Exchange ecosystem. Given 141 natural-language queries, the system ranks **216,041 technical documents** from five communities — *Android, Gaming, Programmers, TeX, Unix* — and returns the top-100 most relevant documents per query.

The system combines:
- **Sparse retrieval** — BM25+ with a tech-aware tokenizer (handles `c++`, `c#`, `.net`, …)
- **Dense retrieval** — semantic embeddings via `all-MiniLM-L12-v2` with SHA-256 caching
- **Fusion** — Reciprocal Rank Fusion (RRF) to merge both signals
- **Classification** — LinearSVC to predict the source Stack Exchange community
- **Reranking** — hard-filter promoting category-matched documents to the top

---

## Pipeline

### Phase 1 — Hybrid Retrieval

```
                          ┌─────────────────────────────┐
                          │        Raw Query            │
                          └──────────────┬──────────────┘
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               │                                                   │
               ▼                                                   ▼
   ┌───────────────────────┐                        ┌───────────────────────────┐
   │      BM25+ Branch     │                        │    Embedding Branch       │
   │                       │                        │                           │
   │  Tech-aware tokenizer │                        │  all-MiniLM-L12-v2        │
   │  (c++, c#, .net, …)   │                        │  384-dim vectors          │
   │  BM25Plus over 216k   │                        │  SHA-256 cache (.npy)     │
   │  docs                 │                        │  Cosine similarity        │
   │                       │                        │  GPU / MPS / CPU auto     │
   └───────────┬───────────┘                        └─────────────┬─────────────┘
               │                                                  │
               │  rank_bm25(d)                        rank_emb(d) │
               │                                                  │
               └─────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │     RRF Fusion               │
                      │                              │
                      │  score(d) =                  │
                      │    w_emb  / (k + rank_emb)   │
                      │  + w_bm25 / (k + rank_bm25)  │
                      │                              │
                      │  w_emb=3.5  w_bm25=0.5  k=20 │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                              Top-100 Documents
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `embedding_model_name` | `all-MiniLM-L12-v2` | SentenceTransformer model |
| `hybrid_rrf_k` | `20` | RRF constant (rank discount) |
| `hybrid_weight_embeddings` | `3.5` | Embedding branch weight |
| `hybrid_weight_bm25` | `0.5` | BM25 branch weight |
| `top_k` | `100` | Final number of returned results |

---

### Phase 2 — Classification & Reranking

```
                          ┌─────────────────────────────┐
                          │        Test Query           │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────┐
                      │  1. Feature Extraction           │
                      │     TF-IDF (~5 000 terms)        │
                      │     fitted on 216k documents     │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  2. LinearSVC Classifier         │
                      │     Macro-F1 = 0.9852            │
                      │     → predicted_category         │
                      │     android | gaming | tex |     │
                      │     unix | programmers           │
                      └──────────────┬───────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
     ┌──────────────────────────┐     ┌──────────────────────────────┐
     │  3. Query Expansion      │     │  4. Pseudo-Relevance         │
     │                          │     │     Feedback (PRF)           │
     │  query += category vocab │     │                              │
     │  e.g. "android" adds:    │     │  BM25 top-3 docs             │
     │  "mobile app java kotlin │     │  → extract tags              │
     │   apk development"       │     │  → append (max 10 tags)      │
     └──────────────────────────┘     └──────────────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  5. Hybrid Retrieval             │
                      │     BM25+ + Embeddings + RRF     │
                      │     Candidate pool = 1 000 docs  │
                      │     (top_k × retrieval_k_mult.)  │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  6. Hard-Filter Reranking        │
                      │     category-match docs → front  │
                      │     relative order preserved     │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                              Top-100 final ranking
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `classifier_method` | `svc` | `svc` \| `logreg` \| `nb` \| `mlp` |
| `feature_method` | `tfidf` | `tfidf` \| `count` \| `embeddings` |
| `rerank_strategy` | `hard_filter` | `hard_filter` \| `soft_boost` |
| `retrieval_k_multiplier` | `10` | Candidate pool = `top_k × 10` |
| `prf_top_k` | `3` | Number of PRF feedback documents |
| `prf_max_tags` | `10` | Max tags injected via PRF |
| `hybrid_rrf_k` | `30` | RRF constant for Phase 2 |

---

## Project Structure

```
.
├── src/                              # Core library
│   ├── config.py                     # Dataclass configs & path auto-resolution
│   ├── data/
│   │   ├── load.py                   # JSON data loader
│   │   └── preprocess.py             # Content field builder & text cleaner
│   ├── retrieval/
│   │   ├── tfidf.py                  # TF-IDF baseline
│   │   ├── bm25.py                   # BM25 / BM25+ with tech tokenizer
│   │   ├── embeddings.py             # Dense retrieval + SHA-256 cache
│   │   └── hybrid.py                 # RRF fusion
│   ├── classification/
│   │   ├── interfaces.py             # Category definitions & protocols
│   │   ├── features.py               # TF-IDF / Count / Embedding feature builders
│   │   ├── model.py                  # Unified classifier (SVC / LogReg / MLP / NB)
│   │   └── rerank.py                 # hard_filter & soft_boost strategies
│   ├── evaluation/
│   │   ├── metrics.py                # Precision@k, Recall@k, MRR@k, F1@k
│   │   └── evaluate.py               # Evaluation pipelines
│   └── kaggle/
│       ├── format.py                 # Kaggle CSV formatter
│       ├── submit_prep.py            # Phase 1 CLI entry point
│       └── submit_phase2.py          # Phase 2 CLI entry point
│
├── notebooks/                        # Step-by-step exploration & ablations
│   ├── 01_data_exploration.ipynb     # EDA, dataset stats
│   ├── 02_tfidf_retrieval.ipynb      # TF-IDF baseline
│   ├── 03_bm25_retrieval.ipynb       # BM25 / BM25+ tuning
│   ├── 04_embeddings_retrieval.ipynb # Dense retrieval + UMAP
│   ├── 05_evaluation_suite.ipynb     # Unified method comparison
│   ├── 06_phase1_submission.ipynb    # ★ Standalone Phase 1 pipeline
│   ├── 07_phase2_classification_ablation.ipynb  # 11-config ablation
│   ├── 08_phase2_pipeline_eval.ipynb # End-to-end Phase 2 evaluation
│   └── 09_phase2_submission.ipynb    # ★ Standalone Phase 2 pipeline
│
├── data/
│   ├── raw/         # Original JSON files (docs, queries, qrels)
│   ├── processed/   # Pre-built content field
│   └── cache/       # Embedding .npy arrays (SHA-256 keyed)
│
├── outputs/
│   ├── runs/        # Evaluation logs (JSON)
│   └── submissions/ # Kaggle submission CSVs
│
└── requirements.txt
```

---

## Getting Started

### Requirements

- Python **3.10+**
- A Kaggle account + API token *(to download the dataset)*
- GPU with CUDA or Apple Silicon *(optional — CPU also works)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/thmsgo18/Information-Retrieval-Engine.git
cd Information-Retrieval-Engine

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate.bat      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Data Setup

```bash
# Configure Kaggle credentials (one-time)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
cd data/raw
kaggle competitions download -c retrieval-engine-competition
unzip retrieval-engine-competition.zip && rm retrieval-engine-competition.zip
cd ../..
```

Expected files in `data/raw/`:
```
docs.json           ← 216,041 Stack Exchange documents
queries_train.json  ← Training queries with ground truth
queries_test.json   ← 141 test queries (Kaggle evaluation)
qgts_train.json     ← Relevance judgments
```

---

## Usage

### Phase 1 — Hybrid Retrieval

**CLI (recommended):**
```bash
# Default production config
python3 -m src.kaggle.submit_prep

# Quick local test (smaller subset + lighter model)
python3 -m src.kaggle.submit_prep \
  --max-docs 20000 \
  --embedding-model-name all-MiniLM-L6-v2 \
  --embedding-precision int8
```

**Notebook:**
```bash
jupyter lab
# Open notebooks/06_phase1_submission.ipynb → Run All
```

Output: `outputs/submissions/submission.csv`

---

### Phase 2 — Classification & Reranking

**CLI (recommended):**
```bash
# Default production config
python3 -m src.kaggle.submit_phase2

# Custom options
python3 -m src.kaggle.submit_phase2 \
  --classifier svc \
  --feature-method tfidf \
  --rerank hard_filter \
  --embedding-model all-MiniLM-L12-v2
```

**Notebook:**
```bash
jupyter lab
# Open notebooks/09_phase2_submission.ipynb → Run All
# Auto-detects local / Kaggle / Colab environment
```

Outputs:
- `outputs/submissions/submission.csv`
- `outputs/runs/phase2/run_YYYYMMDD_HHMMSS.json`

---

### Notebooks Overview

| Notebook | Description |
|---|---|
| `01_data_exploration` | Dataset statistics, community distribution, document lengths |
| `02_tfidf_retrieval` | TF-IDF baseline: indexing, retrieval, evaluation |
| `03_bm25_retrieval` | BM25 / BM25+ tuning and ablation |
| `04_embeddings_retrieval` | Dense retrieval with SentenceTransformers + UMAP/t-SNE |
| `05_evaluation_suite` | Precision@k, Recall@k, MRR@k across all methods |
| `06_phase1_submission` | ★ End-to-end Phase 1 Kaggle submission |
| `07_phase2_classification_ablation` | 11-config ablation (3 feature types × 4 classifiers) |
| `08_phase2_pipeline_eval` | Full Phase 2 evaluation & error analysis |
| `09_phase2_submission` | ★ Final Phase 2 submission (Kaggle/Colab/local) |

**Recommended order:** `01 → 02 → 03 → 04 → 05 → 06` (Phase 1), then `07 → 08 → 09` (Phase 2).

---

## Configuration

All hyperparameters are centralized in [`src/config.py`](src/config.py) as Python dataclasses.

**Phase 1 — `Phase1HybridSubmissionConfig`:**

```python
embedding_model_name: str = "all-MiniLM-L12-v2"
embedding_batch_size: int = 64
embedding_device: str | None = "auto"       # auto → cuda / mps / cpu
embedding_precision: str = "float32"        # float32 | int8 | uint8 | binary
hybrid_bm25_method: str = "plus"            # plus | okapi
hybrid_rrf_k: int = 20
hybrid_weight_embeddings: float = 3.5
hybrid_weight_bm25: float = 0.5
top_k: int = 100
max_docs: int | None = None                 # None = full corpus
```

**Phase 2 — `Phase2Config`:**

```python
classifier_method: str = "svc"              # svc | logreg | nb | mlp
feature_method: str = "tfidf"              # tfidf | count | embeddings
rerank_strategy: str = "hard_filter"       # hard_filter | soft_boost
retrieval_k_multiplier: int = 10           # candidate pool = top_k × 10
prf_top_k: int = 3
prf_max_tags: int = 10
hybrid_rrf_k: int = 30
hybrid_weight_embeddings: float = 3.5
hybrid_weight_bm25: float = 0.5
random_seed: int = 42
```

---

## Results

### Kaggle Leaderboard

| Phase | Configuration | Score |
|---|---|---|
| Phase 2 | SVC + hard_filter, RETRIEVAL_K=200, rrf_k=20 | 0.450 |
| Phase 2 | SVC + hard_filter, RETRIEVAL_K=500, rrf_k=60 | 0.469 |
| Phase 2 | SVC + hard_filter + cross-encoder, RETRIEVAL_K=1000 | 0.400 |
| **Phase 2** | **SVC + hard_filter, RETRIEVAL_K=1000, rrf_k=30** | **0.49541 ✅** |

### Classifier Ablation (notebook 07)

| Features | Classifier | Macro-F1 |
|---|---|---|
| **TF-IDF** | **LinearSVC** | **0.9852 ✅** |
| TF-IDF | LogisticRegression | 0.9794 |
| Count | LinearSVC | 0.9793 |
| Embeddings | MLP | 0.9695 |

### Per-Category Performance (best model)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| android | ~0.99 | ~0.99 | ~0.99 |
| gaming | ~0.97 | ~0.98 | ~0.97 |
| programmers | ~0.99 | ~0.98 | ~0.98 |
| tex | ~1.00 | ~1.00 | ~1.00 |
| unix | ~0.99 | ~0.98 | ~0.99 |
| **Macro avg** | | | **0.9852** |

---

## Authors

[Clara Ait Mokhtar](https://github.com/claraait123), [Maria Aydin](https://github.com/Mmajora53), [Vincent Tan](https://github.com/20centan), [Thomas Gourmelen](https://github.com/thmsgo18)

---

<div align="center">

Master VMI and Master IAD — Data Science Project 2025-2026 &nbsp;·&nbsp; Université Paris

</div>
