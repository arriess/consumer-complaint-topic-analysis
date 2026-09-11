# Consumer Complaint Topic Analysis

Final reproducible implementation for the IU **DLBDSEDA02 - Project: Data Analysis, Task 1** portfolio.

## Project objective

The project uses NLP to identify recurring subthemes in unstructured complaint narratives so that frequently raised concerns can be summarized without manually reading every record. The practical success criterion is not a single model score: the workflow must produce reproducible, interpretable recurring themes and transparent document-assignment shares while keeping the limits of the sample explicit.

## Actual dataset scope

The selected public `complaints_sample.csv` contains 5,000 CFPB complaints, all from the product category **Credit reporting, credit repair services, or other personal consumer reports** and the issue **Incorrect information on your report**. The project therefore analyzes recurring subthemes within incorrect credit-reporting complaints rather than representing all CFPB complaints.

## Implemented workflow

1. Validate the 5,000-row / 18-column public sample.
2. Remove missing and duplicate complaint narratives.
3. Clean and tokenize complaint text.
4. Build Bag-of-Words and TF-IDF matrices.
5. Fit LDA on Bag of Words.
6. Fit LSA (`TruncatedSVD`) on TF-IDF.
7. Compare candidate topic counts 4-8.
8. Extract top terms and representative complaints.
9. Calculate topic/component assignment shares.
10. Generate diagnostic and prevalence visualizations.
11. Optionally stress-test the selected topic structures across multiple random seeds.

## Why these methods

- **Bag of Words -> LDA:** LDA is a probabilistic count-based topic model, so raw term counts provide the natural input while retaining direct interpretability.
- **TF-IDF -> LSA:** TF-IDF down-weights corpus-wide vocabulary and emphasizes discriminative terms before truncated SVD extracts latent semantic directions.
- **Comparable vocabulary:** both representations use the same `min_df`, `max_df`, feature cap and unigram/bigram range so differences are less confounded by vocabulary construction.
- **Multiple diagnostics:** NPMI coherence and topic diversity are compared across both methods, while LDA perplexity and LSA cumulative explained variance are retained as method-specific diagnostics. Final interpretation also considers top terms and representative documents rather than optimizing one metric in isolation.

## Executed results

- Raw rows: **5,000**
- Exact duplicate narratives removed: **683**
- Documents modeled: **4,316**
- Vector vocabulary: **5,000 unigram/bigram features**
- Matrix sparsity: **98.82%**
- Selected LDA solution: **8 topics**, NPMI **0.2411**
- Selected LSA solution: **4 components**, NPMI **0.3421**
- Largest LDA topic assignment share: **53.82%**

The 8-topic LDA solution provides more granular, probabilistic themes. The 4-component LSA solution provides broader latent semantic dimensions and achieved higher NPMI coherence in the tested candidates. LSA component shares are descriptive assignments based on the largest absolute component loading, not probabilities.

See `RESULTS_SUMMARY.md` for the complete executed results and `REPRODUCIBILITY.md` for the independently re-run verification and multi-seed stress test.

## Project structure

```text
consumer-complaint-topic-analysis/
├── README.md
├── FINAL_PRODUCT.md
├── REPRODUCIBILITY.md
├── RESULTS_SUMMARY.md
├── PHASE2_IMPLEMENTATION_NOTES.md
├── PHASE2_REFLECTION.md
├── requirements.txt
├── requirements-lock.txt
├── run_analysis.py
├── run_phase2.py
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
│   │   └── README.md
│   └── tables/
│       └── stability_check.csv
└── src/
    ├── 01_acquire_validate.py
    ├── 02_preprocess.py
    ├── 03_vectorize_and_model.py
    └── 04_stability_check.py
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the exact environment used in the final verification, install `requirements-lock.txt` instead.

## Run the complete analysis

```bash
python run_analysis.py
```

The runner executes, in order:

```bash
python src/01_acquire_validate.py
python src/02_preprocess.py
python src/03_vectorize_and_model.py
```

Optional multi-seed robustness check after the core run:

```bash
python src/04_stability_check.py
```

`run_phase2.py` is retained as the historical development-phase runner and executes the same three core stages.

If the selected dataset is not already present, the acquisition script downloads the exact public `complaints_sample.csv` snapshot automatically. The dataset SHA-256 used in the final verification is recorded in `data/README.md` and `REPRODUCIBILITY.md`.

## Preprocessing

The executable preprocessing follows the submitted Phase 1 plan:

- lowercase
- remove URLs and e-mail artefacts
- remove CFPB-style `XX` / `XXXX` placeholders
- remove punctuation and numeric noise
- regex tokenization
- remove scikit-learn English stop words

## Modeling settings

Both vectorizers use:

- `min_df=5`
- `max_df=0.95`
- `max_features=5000`
- `ngram_range=(1, 2)`

TF-IDF additionally uses `sublinear_tf=True`.

LDA uses online learning, `max_iter=5`, `batch_size=256`, and `random_state=42`. LSA uses `TruncatedSVD` with `random_state=42`.

Candidate counts: **4, 5, 6, 7, 8**.

## Outputs and interpretation

The automated pipeline generates data-validation statistics, vectorization comparison, model diagnostics, topic terms, representative complaints, topic/component assignment shares, selected-model metadata, and diagnostic/prevalence figures.

`topic_labels.csv` and `topic_prevalence_labeled.csv` are human-interpreted presentation tables created after inspecting the automated top terms, representative complaints, and assignment shares. They are intentionally distinguished from algorithmically generated outputs.

Generated source data, representative-complaint exports, and PNG figures are not committed by default; they are recreated by `python run_analysis.py`. The committed reference tables provide compact evidence of the executed results.

## Reliability, maintainability and scalability

- **Reliability:** the workflow validates the expected data structure, fixes stochastic seeds for the submitted models, records exact dependencies, stores a source-file checksum, and includes an independent rerun plus multi-seed robustness check.
- **Maintainability:** acquisition/validation, preprocessing, modeling and optional stability analysis are separated into clear stages with a single neutral runner for the core workflow and documented outputs.
- **Scalability:** sparse document-term matrices and a 5,000-feature cap keep memory use controlled for the selected corpus. The project does **not** claim large-scale production scalability because performance was verified only on the 4,316 modeled documents.

## Reproducibility verification

Before Phase 3 submission, an independent rerun reproduced the core counts and model diagnostics, including the 4,316 modeled documents, 98.82% matrix sparsity, LDA NPMI 0.2411 at 8 topics, LSA NPMI 0.3421 at 4 components, and the 53.82% largest LDA assignment share.

A five-seed stress test additionally showed that 8-topic LDA retained a small mean NPMI advantage over 6-topic LDA (**0.2184 vs. 0.2156**) and lower mean perplexity (**780.59 vs. 858.08**), while LSA-4 reproduced the same diagnostics for all tested seeds. This supports the chosen LDA granularity but also demonstrates why the result should not be presented as uniquely optimal. Exact package versions, the source-file hash, and the complete stability table are documented in `REPRODUCIBILITY.md` and `outputs/tables/stability_check.csv`.

## Limitations

The public sample is deliberately narrow, so conclusions are limited to recurring subthemes within incorrect credit-reporting complaints. Exact duplicates are removed, but near-duplicate legal or dispute templates can still influence the learned topic structure. LDA is also seed-sensitive, reinforcing the decision to interpret the 8-topic solution as a defensible analytical view rather than a unique ground truth.
