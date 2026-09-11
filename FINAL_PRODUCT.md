# Final Product - Consumer Complaint Topic Analysis

## Repository

https://github.com/arriess/consumer-complaint-topic-analysis

## Purpose

This repository contains the final reproducible NLP workflow for IU DLBDSEDA02 Task 1. It validates and preprocesses a public CFPB complaint sample, compares Bag of Words with TF-IDF, compares LDA with LSA/TruncatedSVD, evaluates candidate topic counts, and exports diagnostics, topic terms, representative complaints, assignment shares, and figures.

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python run_analysis.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
python run_analysis.py
```

For a compatible rather than exact environment, `requirements.txt` contains bounded version ranges.

The acquisition step downloads the selected public `complaints_sample.csv` snapshot automatically when it is not already present. The verified source-file SHA-256 is documented in `data/README.md` and `REPRODUCIBILITY.md`.

## Pipeline stages

1. `src/01_acquire_validate.py` - download/validate the sample and remove empty/duplicate narratives.
2. `src/02_preprocess.py` - clean and tokenize text using the submitted Phase 1 preprocessing plan.
3. `src/03_vectorize_and_model.py` - create BoW/TF-IDF representations, fit LDA/LSA candidates, evaluate models, and generate outputs.
4. Optional: `src/04_stability_check.py` - stress-test the selected LDA/LSA structures across five random seeds.

`run_phase2.py` is retained as the historical development-phase runner; `run_analysis.py` is the neutral final entry point for the three core stages.

## Automated outputs

The pipeline generates validation statistics, vectorization comparison, model diagnostics, topic terms, representative complaints, topic/component assignment shares, selected-model metadata, and diagnostic/prevalence figures.

`topic_labels.csv` and `topic_prevalence_labeled.csv` are human-interpreted presentation tables created after inspecting the automated top terms, representative complaints, and assignment shares. This distinction prevents interpretive labels from being presented as automatically generated model output.

Generated source data, representative-complaint exports, and PNG figures are excluded from Git by default and are recreated by running the pipeline. Compact reference result tables remain committed for review.

## Reproducibility and robustness

Before Phase 3 submission, the core workflow was independently re-run using the exact versions in `requirements-lock.txt`. The rerun reproduced the main dataset counts, vectorization dimensions, selected LDA/LSA diagnostics, and topic-assignment shares documented in `RESULTS_SUMMARY.md`.

An additional five-seed stress test is committed in `outputs/tables/stability_check.csv`. It showed that 8-topic LDA retained a small mean NPMI advantage over 6-topic LDA and lower mean perplexity, while LSA-4 reproduced identical diagnostics across the tested seeds. This supports the submitted models while documenting LDA seed sensitivity rather than hiding it.

Run the optional robustness check after the core pipeline with:

```bash
python src/04_stability_check.py
```

Full verification details, exact package versions and the dataset checksum are in `REPRODUCIBILITY.md`.
