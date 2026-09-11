# Final Product - Consumer Complaint Topic Analysis

## Repository

https://github.com/arriess/consumer-complaint-topic-analysis

## Purpose

This repository contains the final reproducible NLP workflow for IU DLBDSEDA02 Task 1. It validates and preprocesses a public CFPB complaint sample, compares Bag of Words with TF-IDF, compares LDA with LSA/TruncatedSVD, evaluates candidate topic counts, and exports diagnostics, interpreted topic information, representative complaints, assignment shares, and figures.

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_phase2.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_phase2.py
```

The acquisition step downloads the exact selected public `complaints_sample.csv` snapshot automatically when it is not already present.

## Pipeline stages

1. `src/01_acquire_validate.py` - download/validate the sample and remove empty/duplicate narratives.
2. `src/02_preprocess.py` - clean and tokenize text using the submitted Phase 1 preprocessing plan.
3. `src/03_vectorize_and_model.py` - create BoW/TF-IDF representations, fit LDA/LSA candidates, evaluate models, and generate outputs.

## Main outputs

The pipeline writes result tables under `outputs/tables/`, including validation statistics, vectorization comparison, model diagnostics, topic terms, topic labels, representative complaints, topic/component assignment shares, and selected-model summaries. It also generates diagnostic and prevalence figures under `outputs/figures/`.

## Reproducibility

The workflow uses a fixed random state of 42 for stochastic model steps and records all Python dependencies in `requirements.txt`. The selected dataset source, vectorization settings, candidate topic counts, diagnostics, and limitations are documented in the repository.
