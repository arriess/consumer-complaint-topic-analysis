# Consumer Complaint Topic Analysis

Final reproducible implementation for the IU **DLBDSEDA02 - Project: Data Analysis, Task 1** portfolio.

## Project objective

The project uses NLP to identify recurring subthemes in unstructured complaint narratives so that frequently raised concerns can be summarized without manually reading every record.

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

See `RESULTS_SUMMARY.md` for the complete executed results and `PHASE2_REFLECTION.md` for the submitted development/reflection summary.

## Project structure

```text
consumer-complaint-topic-analysis/
├── README.md
├── FINAL_PRODUCT.md
├── RESULTS_SUMMARY.md
├── PHASE2_IMPLEMENTATION_NOTES.md
├── PHASE2_REFLECTION.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   ├── figures/   # generated when the pipeline runs
│   └── tables/
├── src/
│   ├── 01_acquire_validate.py
│   ├── 02_preprocess.py
│   └── 03_vectorize_and_model.py
└── run_phase2.py
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

## Run the complete analysis

```bash
python run_phase2.py
```

The runner executes, in order:

```bash
python src/01_acquire_validate.py
python src/02_preprocess.py
python src/03_vectorize_and_model.py
```

If the selected dataset is not already present, the acquisition script downloads the exact public `complaints_sample.csv` snapshot automatically.

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

## Generated outputs

Tables include data-validation statistics, vectorization comparison, model diagnostics, top topic terms, topic labels, representative complaints, assignment shares, and selected-model summaries. The pipeline also generates diagnostic and prevalence figures in `outputs/figures/`.

## Limitations

The public sample is deliberately narrow, so conclusions are limited to recurring subthemes within incorrect credit-reporting complaints. Exact duplicates are removed, but near-duplicate legal or dispute templates can still influence the learned topic structure. The 8-topic LDA choice should also be interpreted cautiously because the 6-topic LDA model had very similar coherence.
