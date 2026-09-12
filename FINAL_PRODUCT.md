# Final Product - Consumer Complaint Topic Analysis

## Repository

https://github.com/arriess/consumer-complaint-topic-analysis

## Purpose

This repository contains the final reproducible NLP workflow for IU DLBDSEDA02 Task 1. It validates and preprocesses a fixed public CFPB complaint sample, compares Bag of Words with TF-IDF, compares LDA with LSA/TruncatedSVD, evaluates candidate topic counts, exports diagnostics and interpretation evidence, and documents limitations and robustness.

## Verified environment

- Python: **3.13.5** (`.python-version`)
- Exact direct/transitive package environment: `requirements-lock.txt`
- Compatible direct-dependency ranges: `requirements.txt`

## Exact reproduction

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

The acquisition stage downloads the selected fixed `complaints_sample.csv` when needed and verifies its SHA-256 before analysis. Official CFPB provenance, the exact technical snapshot URL, scope, and checksum are documented in `data/README.md`.

## Pipeline stages

1. `src/01_acquire_validate.py` - checksum-validate the source, validate structure/scope, remove empty and exact-duplicate narratives.
2. `src/02_preprocess.py` - clean and tokenize text using the submitted Phase 1 preprocessing plan.
3. `src/03_vectorize_and_model.py` - create BoW/TF-IDF representations, fit LDA/LSA candidates, evaluate models, and generate outputs.
4. Optional: `src/04_stability_check.py` - stress-test the selected LDA/LSA structures across five random seeds.

`run_analysis.py` is the final core entry point. `run_phase2.py` is retained only as the historical development-phase runner and executes the same three core stages.

## Expected verification checkpoints

A correct run should reproduce the submitted fixed-seed analysis within normal deterministic library behavior:

- source: **5,000 rows x 18 columns**
- exact duplicate narratives removed: **683**
- final modeled documents: **4,316**
- vector matrices: **4,316 x 5,000**
- matrix sparsity: **98.82%**
- LDA-8: NPMI **0.2411**, diversity **0.6375**, perplexity **759.25**
- LSA-4: NPMI **0.3421**, diversity **0.8000**, explained variance **4.81%**
- largest LDA document-assignment share: **53.82%**

## Outputs and interpretation

The pipeline generates data validation, vectorization comparison, model diagnostics, topic terms, representative complaints, topic/component assignment shares, selected-model metadata, and figures.

Compact reference tables are committed in `outputs/tables/`. Raw source/intermediate data, representative-complaint narrative exports, and PNG figures are recreated locally and are not committed by default.

`topic_labels.csv` and `topic_prevalence_labeled.csv` are human-interpreted presentation tables, not automatically generated class labels. `INTERPRETATION_NOTES.md` documents the top-term evidence behind each label. LSA assignment shares use the largest absolute component loading and are explicitly **not probabilities**.

## Robustness check

After the core run, execute:

```bash
python src/04_stability_check.py
```

This writes `outputs/tables/stability_check.csv`. Across seeds 7, 21, 42, 84, and 123, LDA k=8 retained a small mean NPMI advantage over k=6 and lower mean perplexity, while LSA-4 reproduced the same diagnostics across the tested seeds. The result supports the submitted models while documenting LDA seed sensitivity instead of hiding it.

## Documentation map

- `README.md` - objective, method rationale, workflow, limitations
- `RESULTS_SUMMARY.md` - executed findings
- `INTERPRETATION_NOTES.md` - evidence behind topic labels
- `REPRODUCIBILITY.md` - environment, checksum, rerun, stability evidence
- `data/README.md` - official CFPB provenance and exact snapshot identity

The repository is intentionally organized so that the analysis can be inspected at two levels: concise final documentation for review and executable source code/machine-readable results for verification.
