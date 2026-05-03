# Cybully — Cyberbullying Detection with DistilBERT

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/AA-Hugger/cybully-classifier)

End-to-end NLP pipeline that fine-tunes DistilBERT on the HateXplain dataset to classify social media posts as **normal**, **offensive**, or **hatespeech**, with a FastAPI backend and a React frontend for interactive predictions.

---

## Overview

Cybully trains a 3-class text classifier on 19,229 real social media posts sourced from HateXplain — a dataset with per-post annotations from multiple human raters. Labels are resolved via majority vote; posts where all three annotators disagreed (919 posts) are held out for post-training ambiguity analysis rather than used for training.

Four model versions are trained incrementally, each improving on the last (class weighting, learning rate tuning, extended epochs). All four are served via a REST API and switchable in the UI, making it easy to compare their behaviour on the same input.

---

## Features

- **Domain-specific tokenizer** — 20 custom special tokens (`<user>`, `<url>`, `<censored>`, emoji tokens, etc.) registered with DistilBERT to prevent subword splitting
- **Class weighting** — inverse-frequency weights to handle the imbalanced label distribution
- **4 model versions** — incrementally improved checkpoints, all downloadable from HuggingFace Hub
- **Ambiguity analysis** — dedicated script analysing model confidence on the 919 posts with no annotator consensus
- **FastAPI backend** — REST API with model switching, CORS support, and static file serving
- **React SPA** — input form, model selector, confidence bar charts, and example posts
- **Docker deployment** — targets HuggingFace Spaces (port 7860)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Training | PyTorch + HuggingFace Transformers |
| API | FastAPI + Uvicorn |
| Frontend | React 19 + Vite + Axios |
| Container | Docker (Python 3.11-slim) |

---

## Project Structure

```
cybully/
├── src/
│   ├── train.py              # Fine-tuning pipeline (DistilBERT + AdamW + early stopping)
│   ├── evaluate.py           # Metrics, classification report, confusion matrix
│   ├── predict.py            # Single-text inference with in-memory model caching
│   ├── preprocess.py         # Text normalisation, tokenisation, special tokens
│   ├── dataset.py            # PyTorch Dataset + stratified train/val/test split
│   └── analyse_ambiguous.py  # Confidence analysis on 3-way annotator disagreements
│
├── api/
│   └── main.py               # FastAPI app — /models, /predict, static file serving
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── PostInput.jsx     # Textarea + model selector
│   │       ├── ResultCard.jsx    # Label badge + confidence bars
│   │       └── ExamplePosts.jsx  # Preloaded example posts
│   └── package.json
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_tokenisation_example.ipynb  # Custom token registration demo
│
├── assets/                   # Generated visualisation PNGs
├── data/                     # Raw + processed data (gitignored)
├── models/                   # Trained checkpoints (gitignored, download via script)
├── download_models.py        # Downloads all 4 models from HuggingFace Hub
├── Dockerfile                # Production container for HuggingFace Spaces
└── requirements.txt
```

---

## Dataset

**HateXplain** — [Kaggle link](https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification/data?select=hateXplain.csv)

| Stat | Value |
|------|-------|
| Total annotations | 60,444 |
| Unique posts | 20,148 |
| Annotators | 253 (~3 per post) |
| Training set | 19,229 posts (majority vote) |
| Ambiguous (held out) | 919 posts (3-way tie) |

**Label distribution (after majority vote):**

| Label | Count | % |
|-------|-------|---|
| Normal | 7,814 | 40.6% |
| Hatespeech | 5,935 | 30.9% |
| Offensive | 5,480 | 28.5% |

Place the raw CSV at `data/raw/hateXplain.csv`. The training script reads from `data/processed/` (written by the preprocessing step).

---

## Models

All four checkpoints are hosted on HuggingFace Hub and downloaded automatically by `download_models.py`.

| Version | Local Folder | Description |
|---------|-------------|-------------|
| Baseline | `models/baseline` | No class weighting, LR 2e-5 |
| Class Weights | `models/v2_class_weights` | Inverse-frequency class weights |
| Lower LR | `models/v3_lower_lr` | LR reduced to 1e-5 |
| Best Model | `models/v4_more_epochs` | 8 epochs + class weights (default) |

**Training hyperparameters (v4):**
- Batch size: 32 | Max tokens: 128 | Warmup ratio: 0.1
- Optimizer: AdamW | Gradient clipping: 1.0
- Early stopping: patience 3 on validation loss

---

## Setup

**Prerequisites:** Python 3.11+, Node.js 18+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cybully.git
cd cybully

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download trained models from HuggingFace Hub
python download_models.py

# 4. Install frontend dependencies
cd frontend && npm install && cd ..
```

---

## Usage

### Train from scratch

```bash
python src/train.py
```

Trains on `data/processed/` and saves the best checkpoint to `models/v4_more_epochs/`.

### Evaluate a model

```bash
python src/evaluate.py
```

Prints accuracy + per-class F1, and saves a confusion matrix to `assets/`.

### Run the API (development)

```bash
uvicorn api.main:app --reload
# http://localhost:8000
```

### Run the frontend (development)

```bash
cd frontend
npm run dev
# http://localhost:5173
```

### Build frontend for production

The FastAPI server serves the built React app as static files. Build before deploying:

```bash
cd frontend && npm run build
```

### Docker (HuggingFace Spaces)

```bash
docker build -t cybully .
docker run -p 7860:7860 cybully
# http://localhost:7860
```

The Dockerfile downloads all four model versions from HuggingFace Hub at build time and serves the API on port 7860.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Label distribution, annotator agreement breakdown, text length analysis, word clouds by class, target group frequency |
| [02_tokenisation_example.ipynb](notebooks/02_tokenisation_example.ipynb) | Demonstrates why domain-specific tokens like `<user>` need special registration and how to do it with HuggingFace tokenizers |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | Returns list of available model names |
| `POST` | `/predict` | Classifies a post |

**POST /predict — request body:**

```json
{
  "text": "your social media post here",
  "model_name": "Best Model"
}
```

**POST /predict — response:**

```json
{
  "label": "hatespeech",
  "confidence": 0.94,
  "scores": {
    "normal": 0.02,
    "offensive": 0.04,
    "hatespeech": 0.94
  },
  "model_used": "Best Model"
}
```

Available `model_name` values: `"Baseline"`, `"Class Weights"`, `"Lower Learning Rate"`, `"Best Model"`

---

## Attribution

Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021. Dataset accessed via [Kaggle](https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification).

> **Note:** The example posts and word cloud visualisations in this project contain offensive and hateful language drawn directly from the dataset. This content is present for research and evaluation purposes only.
