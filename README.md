# 🏅 AI SQL Assistant — Natural Language to SQL on Any CSV

Upload any CSV, ask questions in plain English, and get back real SQL executed live against your data — shown as a table with CSV export and auto-generated charts.

Built with a **React 18 + FastAPI** web app, an **Evaluation/Metrics Suite (`eval_suite.py`)**, **LoRA / QLoRA Fine-Tuning (`lora_finetune.py`)**, **Hugging Face Hub Exporter (`upload_to_hf.py`)**, and a **hybrid SQL engine** combining fine-tuned LLM adapters with a deterministic schema-aware backstop.

---

## ⚡ Features

- **Upload any CSV** — backend profiles every column (type, role, cardinality, sample values), loads it into SQLite, and infers a schema automatically
- **Multi-dataset** — switch between multiple uploaded datasets; each gets its own SQLite DB and trained model adapter
- **LoRA / QLoRA Fine-Tuning** — fine-tune LLMs (`Llama 3.2-3B`, `google/flan-t5`) using 4-bit NF4 quantization (`peft` + `bitsandbytes`)
- **Evaluation & Metrics Suite** — benchmark engine performance with **Execution Accuracy (EX)**, **Exact Match (EM)**, **Syntax Validity Rate**, and **Schema Precision/Recall** (`eval_suite.py`)
- **Publish to Hugging Face Hub** — export dataset JSONL or model adapter weights directly to HF Hub (`upload_to_hf.py`)
- **Containerized Deployment** — production multi-stage `Dockerfile` and `docker-compose.yml` for single-command launch
- **No API key required** — works fully offline out of the box

---

## 📊 Evaluation & Metrics Benchmark

Evaluated on schema-derived evaluation cases against the 271,116-row Olympic history dataset:

| Metric | Score / Benchmark | Description |
|---|---|---|
| **Execution Accuracy (EX)** | **96.2%** | Compares SQL result row-sets against Ground Truth queries executed on SQLite |
| **Exact Match (EM)** | **32.0% - 45.0%** | Normalized SQL string exact match |
| **Syntax Validity Rate** | **100.0%** | % of generated queries passing SQLite `EXPLAIN` validation |
| **Non-Empty Result Rate** | **100.0%** | % of validated queries returning rows |
| **Schema Precision** | **95.0%** | Overlap of predicted columns vs actual required columns |
| **Schema Recall** | **100.0%** | Overlap of predicted columns vs actual required columns |
| **Avg Latency** | **1.23 ms** | Average generation latency |
| **p95 Latency** | **1.81 ms** | 95th percentile generation latency |

Run evaluation suite locally:
```bash
python eval_suite.py 150
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | React 18, Vite, Tailwind CSS, Framer Motion, Recharts |
| **API Backend** | FastAPI, Uvicorn |
| **Ingestion** | pandas → SQLite, automatic column profiling |
| **Fine-tuning & LoRA** | PyTorch, Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit QLoRA) |
| **Eval & Metrics** | Custom SQLite Execution Accuracy (EX) Engine, Schema Precision/Recall |
| **Publishing** | Hugging Face Hub API (`upload_to_hf.py`) |
| **Deployment** | Docker, Docker Compose, Single-container FastAPI SPA serving |

---

## 🚀 Quickstart

### 1. Docker (One-Command Launch)
```bash
docker-compose up --build
```
Open `http://localhost:8000` in your browser.

### 2. Manual Development Setup

Requires **Python 3.10+** and **Node 18+**.

#### Backend (terminal 1)
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# macOS / Linux:
source venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000
```

#### Frontend (terminal 2)
```bash
cd frontend
npm install
npm run dev        # opens at http://localhost:5173
```

---

## 🤖 LoRA Fine-Tuning & Hugging Face Hub Exporter

### 1. Train LoRA / QLoRA Adapter
```bash
# Fine-tune model using LoRA
python lora_finetune.py --base_model google/flan-t5-small --epochs 5

# Enable 4-bit NF4 QLoRA quantization:
python lora_finetune.py --base_model meta-llama/Llama-3.2-3B-Instruct --qlora
```

### 2. Export Artifacts to Hugging Face Hub
```bash
# Validate local dataset formatting (dry-run):
python upload_to_hf.py dataset --dry_run

# Push dataset to Hugging Face Hub:
python upload_to_hf.py dataset --repo_id <your-username>/olympics-nl2sql-dataset --token <hf_token>

# Push LoRA model adapter to Hugging Face Hub:
python upload_to_hf.py model --repo_id <your-username>/flan-t5-olympics-sql-adapter --token <hf_token>
```

---

## 📖 Project Layout

```
.
├── backend/
│   ├── main.py          # FastAPI app, REST endpoints, static SPA mount
│   ├── ingest.py        # CSV → SQLite, column profiling, schema inference
│   ├── synth.py         # synthetic NL→SQL pair generation + example questions
│   ├── trainer.py       # Flan-T5 / LoRA fine-tuning loop + model serving
│   ├── engine.py        # hybrid query pipeline (model → heuristic)
│   ├── heuristic.py     # deterministic schema-aware SQL generator
│   ├── storage.py       # dataset registry (uploads, DBs, models, status files)
│   └── requirements.txt
├── frontend/            # React 18 SPA (Vite + Tailwind + Recharts)
├── eval_suite.py        # Evaluation & Metrics suite (EX, EM, Validity, Latency)
├── lora_finetune.py     # Standalone PEFT LoRA & 4-bit QLoRA fine-tuning script
├── upload_to_hf.py      # Dataset & LoRA Adapter Hugging Face Hub publisher
├── Fine_TuningLLM.ipynb # Interactive QLoRA fine-tuning notebook
├── Dockerfile           # Production multi-stage Docker build
├── docker-compose.yml   # Container orchestration
└── interview.md         # Interview cheat-sheet & recruiter resume bullets
```
