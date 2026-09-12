# Consumer Complaint Topic Analysis

Final reproducible implementation for the IU **DLBDSEDA02 - Project: Data Analysis, Task 1** portfolio.

## Project objective

The project uses NLP to identify recurring subthemes in unstructured complaint narratives so that frequently raised concerns can be summarized without manually reading every record. The practical success criterion is not a single model score: the workflow must produce reproducible, interpretable recurring themes and transparent document-assignment shares while keeping the limits of the sample explicit.

## Data source and scope

The complaint records originate from the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**. To make the analysis reproducible, the project uses the fixed public `complaints_sample.csv` snapshot selected in Phase 1 rather than a changing live export.

- Official CFPB database: https://www.consumerfinance.gov/data-research/consumer-complaints/
- Exact technical snapshot and SHA-256: documented in `data/README.md`

The verified snapshot contains **5,000 rows and 18 columns**. Every record belongs to the product category **Credit reporting, credit repair services, or other personal consumer reports** and the issue **Incorrect information on your report**. Seven structured sub-issue categories are present. The project therefore analyzes recurring **subthemes within incorrect credit-reporting complaints** rather than representing all CFPB complaints or all consumers.

## Implemented workflow

1. Acquire and checksum-validate the selected 5,000-row snapshot.
2. Validate structure and document dataset scope.
3. Remove missing and exact-duplicate complaint narratives.
4. Clean and tokenize complaint text.
5. Build Bag-of-Words and TF-IDF matrices using comparable vocabulary settings.
6. Fit LDA on Bag of Words.
7. Fit LSA (`TruncatedSVD`) on TF-IDF.
8. Compare candidate topic counts 4-8.
9. Extract top terms and representative complaints.
10. Calculate topic/component document-assignment shares.
11. Generate diagnostic and prevalence visualizations.
12. Optionally stress-test the selected topic structures across multiple random seeds.

## Why these methods and parameters

- **Bag of Words -> LDA:** LDA is a probabilistic topic model designed for count-like word occurrence data, so raw term counts are a natural and interpretable input.
- **TF-IDF -> LSA:** TF-IDF down-weights vocabulary that occurs across much of the corpus before truncated SVD extracts latent semantic directions.
- **Comparable vocabulary:** both representations use the same `min_df`, `max_df`, feature cap, and unigram/bigram range so differences are less confounded by vocabulary construction.
- **`min_df=5`:** filters extremely rare terms that provide little corpus-level evidence and can add noise.
- **`max_df=0.95`:** removes terms appearing in almost every document, where they contribute little discrimination.
- **`max_features=5000`:** keeps the sparse matrices computationally controlled while retaining a broad vocabulary for this 4,316-document corpus.
- **unigrams + bigrams:** preserve both individual terms and recurring short expressions such as `credit report` or `identity theft`.
- **candidate counts 4-8:** provide a compact range from broad to more granular structures for this deliberately narrow corpus. The final count is selected from the executed diagnostics rather than assumed in advance.
- **multiple diagnostics:** NPMI coherence and topic diversity are compared for both methods; LDA perplexity and LSA cumulative explained variance are retained as method-specific diagnostics. Top terms and representative complaints are also inspected so selection is not reduced to one number.

## Executed results

- Raw rows: **5,000**
- Exact duplicate narratives removed: **683**
- Documents after de-duplication: **4,317**
- Documents becoming empty after preprocessing: **1**
- Documents modeled: **4,316**
- Vector vocabulary: **5,000 unigram/bigram features**
- Matrix sparsity: **98.82%**
- Selected LDA solution: **8 topics**, NPMI **0.2411**, diversity **0.6375**, perplexity **759.25**
- Selected LSA solution: **4 components**, NPMI **0.3421**, diversity **0.8000**, cumulative explained variance **4.81%**
- Largest LDA topic assignment share: **53.82%**

The 8-topic LDA solution provides the more granular probabilistic view. The 4-component LSA solution provides broader latent semantic dimensions and achieved higher NPMI coherence among its tested candidates. LSA component shares are descriptive assignments based on the largest absolute component loading, **not probabilities**.

The 6-topic LDA solution had very similar fixed-seed coherence (**0.2390**), so the 8-topic result is treated as a defensible granular view rather than a uniquely optimal ground truth.

See:

- `RESULTS_SUMMARY.md` - executed findings and method comparison
- `INTERPRETATION_NOTES.md` - evidence supporting the human topic labels
- `REPRODUCIBILITY.md` - environment, checksum, independent rerun, and multi-seed robustness check

## Project structure

```text
consumer-complaint-topic-analysis/
├── README.md
├── FINAL_PRODUCT.md
├── REPRODUCIBILITY.md
├── RESULTS_SUMMARY.md
├── INTERPRETATION_NOTES.md
├── PHASE2_IMPLEMENTATION_NOTES.md
├── PHASE2_REFLECTION.md
├── requirements.txt
├── requirements-lock.txt
├── .python-version
├── run_analysis.py
├── run_phase2.py
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
│   │   └── README.md
│   └── tables/
│       ├── data_validation_summary.json
│       ├── vectorization_comparison.csv
│       ├── model_diagnostics.csv
│       ├── selected_model_summary.json
│       ├── topic_terms.csv
│       ├── topic_prevalence.csv
│       ├── topic_labels.csv
│       ├── topic_prevalence_labeled.csv
│       └── stability_check.csv
└── src/
    ├── 01_acquire_validate.py
    ├── 02_preprocess.py
    ├── 03_vectorize_and_model.py
    └── 04_stability_check.py
```

## Reproduce the submitted analysis

The final verification used **Python 3.13.5**, recorded in `.python-version`. `requirements-lock.txt` pins the full direct/transitive Python package environment used for verification. `requirements.txt` contains bounded direct-dependency ranges for a compatible environment.

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

The core runner executes, in order:

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

The acquisition stage verifies the source SHA-256 before analysis and stops on a mismatch. This prevents a changed local or remote snapshot from being analyzed silently.

## Preprocessing

The executable preprocessing follows the submitted Phase 1 plan:

- lowercase
- remove URLs and e-mail artefacts
- remove CFPB-style `XX` / `XXXX` placeholders
- remove punctuation and numeric noise
- deterministic regex tokenization
- remove scikit-learn English stop words
- remove one-character tokens

No lemmatization is claimed or performed in the submitted pipeline.

## Modeling settings

Both vectorizers use:

- `min_df=5`
- `max_df=0.95`
- `max_features=5000`
- `ngram_range=(1, 2)`

TF-IDF additionally uses `sublinear_tf=True`.

LDA uses online learning, `max_iter=5`, `batch_size=256`, and `random_state=42`. LSA uses randomized `TruncatedSVD`, `n_iter=10`, and `random_state=42`.

Candidate topic/component counts: **4, 5, 6, 7, 8**.

## Outputs and interpretation

The automated pipeline generates:

- dataset validation statistics and source identity
- BoW/TF-IDF comparison
- model diagnostics across candidate counts
- selected-model metadata
- top terms
- representative complaints (local generated file)
- topic/component document assignments
- diagnostic and prevalence figures

`topic_labels.csv` and `topic_prevalence_labeled.csv` are **human-interpreted presentation tables** created after inspecting automated top terms, representative complaints, and assignment shares. They are intentionally distinguished from algorithmically generated outputs. `INTERPRETATION_NOTES.md` documents the top-term evidence behind every label.

Generated source/intermediate CSV files, representative-complaint exports, and PNG figures are not committed. The compact reference result tables are committed so a tutor can inspect the executed results without exposing a public narrative export; running the pipeline recreates all generated artifacts.

## Reliability, maintainability, and scalability

- **Reliability:** checksum validation, expected-shape checks, explicit scope validation, fixed submitted seed, fully pinned verification environment, independent rerun, and a multi-seed stress test.
- **Maintainability:** acquisition/validation, preprocessing, modeling, and optional robustness analysis are separated into documented stages; the core workflow has one neutral runner and compact machine-readable outputs.
- **Scalability:** sparse document-term matrices and a 5,000-feature cap control memory use for the selected corpus. The project does **not** claim production-scale scalability because performance was verified only on the 4,316 modeled documents.

## Reproducibility and robustness

The independent rerun reproduced the main dataset counts, vector dimensions, fixed-seed diagnostics, and assignment shares. A five-seed stress test (7, 21, 42, 84, 123) additionally showed:

- LDA k=6 mean NPMI: **0.2156**; mean perplexity: **858.08**; mean diversity: **0.6767**
- LDA k=8 mean NPMI: **0.2184**; mean perplexity: **780.59**; mean diversity: **0.6375**
- LSA k=4 reproduced NPMI **0.3421**, diversity **0.8000**, and explained variance **4.81%** across all tested seeds

This supports retaining the submitted 8-topic LDA as a granular solution while making its seed sensitivity explicit.

## Limitations

The selected sample is deliberately narrow. Conclusions are limited to recurring subthemes within incorrect credit-reporting complaints and should not be generalized to all CFPB complaints or all consumers. Exact duplicates are removed, but near-duplicate legal/dispute templates remain and can influence topic structure. The topic labels are interpretive summaries of learned language patterns rather than verified classifications of individual complaints. LDA seed sensitivity further reinforces that the 8-topic solution is one defensible analytical view rather than unique ground truth.
