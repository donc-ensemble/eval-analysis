# Multi-Framework RAG Evaluation Pipeline

Runs a fixed set of RAG prompt/context/expected-answer triples through **RAGAS**, **Promptfoo**, and **LangSmith** simultaneously, then generates an interactive HTML analytics report synthesized by a local LLM judge.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- [Ollama](https://ollama.com/download) installed and running locally
- [Node.js 18+](https://nodejs.org/) (required by Promptfoo)
- A LangSmith API key ([get one free](https://smith.langchain.com))

---

## Setup

### 1. Install Python dependencies

```bash
uv pip install -r requirements.txt
```

Or with pip and a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull the required Ollama models

Make sure Ollama is running (`ollama serve` in a separate terminal if it isn't), then:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

`llama3` is ~4.7 GB — this only needs to happen once.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual LangSmith API key:

```
LANGSMITH_API_KEY=lsv2_pt_your_key_here
```

---

## Running the Evaluations

Make sure Ollama is running before executing any of the commands below.

### Run all frameworks

```bash
python run.py --framework all --model llama3
```

### Run a single framework

```bash
python run.py --framework ragas     --model llama3
python run.py --framework promptfoo --model llama3
python run.py --framework langsmith --model llama3
```

Results are written incrementally to `data/outputs/<timestamp>/` as the run progresses — a crash mid-run won't lose completed samples.

---

## Generating the HTML Report

After a run completes, generate the analytics report from the latest output:

```bash
python generate_report.py --model llama3
```

Then open it in your browser:

```bash
open data/outputs/$(ls -t data/outputs | head -1)/report.html
```

---

## Output Files

Each run creates a timestamped directory under `data/outputs/`:

```
data/outputs/<timestamp>/
├── ragas.json          # Per-sample RAGAS scores
├── promptfoo.json      # Per-sample Promptfoo scores
├── langsmith.json      # Per-sample LangSmith scores
├── all.json            # All frameworks merged (written when >1 framework ran)
├── summary_report.json # Macro averages + run metadata
└── report.html         # Interactive analytics dashboard
```

---

## Evaluation Metrics

Each framework scores every sample on four dimensions:

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |
| **Answer Relevance** | Does the answer directly address the question? |
| **Context Recall** | Does the retrieved context cover everything in the ground-truth reference? |
| **Answer Correctness** | How factually aligned is the answer with the ground truth? |

---

## Architecture

```
.
├── .env                        # API keys (never commit — see .env.example)
├── .env.example                # Safe template to share with teammates
├── requirements.txt            # Python dependencies
├── run.py                      # CLI entry point
├── generate_report.py          # HTML report generator with LLM analysis
├── data/
│   ├── dataset.json            # Evaluation dataset (question/context/answer/reference)
│   └── outputs/                # Generated run artifacts (git-ignored)
└── src/
    ├── orchestrator.py         # Coordinates runners, writes crash-safe incremental output
    ├── dataset_loader.py
    └── evaluators/
        ├── base.py
        ├── ragas_runner.py     # RAGAS via local Ollama
        ├── promptfoo_runner.py # Promptfoo CLI via npx
        └── langsmith_runner.py # LangSmith SDK + local Ollama judge
```
