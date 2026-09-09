# Consumer Complaint Topic Analysis — Phase 2

Reproducible implementation for the IU **DLBDSEDA02 – Project: Data Analysis, Task 1**.

## Actual dataset scope

The selected public `complaints_sample.csv` contains 5,000 CFPB complaints, all from the product category **Credit reporting, credit repair services, or other personal consumer reports** and the issue **Incorrect information on your report**. The project therefore analyzes recurring subthemes within incorrect credit-reporting complaints.

## Implemented workflow

1. Validate the 5,000-row / 18-column sample.
2. Remove missing and duplicate complaint narratives.
3. Clean and tokenize complaint text.
4. Build Bag-of-Words and TF-IDF matrices.
5. Fit LDA on Bag of Words.
6. Fit LSA (`TruncatedSVD`) on TF-IDF.
7. Compare candidate topic counts 4–8.
8. Extract top terms and representative complaints.
9. Calculate topic/component assignment shares.
10. Generate diagnostic and prevalence visualizations.

## Executed data results

- Raw rows: **5,000**
- Exact duplicate narratives removed: **683**
- Documents modeled: **4,316**
- Vector vocabulary: **5,000 unigram/bigram features**
- Matrix sparsity: **98.82%**
- Selected LDA solution: **8 topics**
- Selected LSA solution: **4 components**

See `RESULTS_SUMMARY.md` for the complete executed results.

## Project structure

```text
consumer-complaint-topic-analysis/
├── README.md
├── RESULTS_SUMMARY.md
├── PHASE2_IMPLEMENTATION_NOTES.md
├── PHASE2_REFLECTION_DRAFT.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
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

## Run the complete pipeline

```bash
python run_phase2.py
```

Or run each step:

```bash
python src/01_acquire_validate.py
python src/02_preprocess.py
python src/03_vectorize_and_model.py
```

The first script downloads the selected public sample if `data/complaints_sample.csv` is not already present.

## Preprocessing

The final executable preprocessing is deliberately self-contained:

- lowercase
- remove URLs and e-mail artefacts
- remove CFPB-style `XX` / `XXXX` placeholders
- remove punctuation and numeric noise
- regex tokenization
- remove scikit-learn English stop words

The Phase 1 conception mentioned WordNet lemmatization. During execution the NLTK corpora could not be retrieved in the environment, so the implementation was changed to avoid an external runtime resource dependency. This change is documented in the Phase 2 reflection.

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

Tables:

- `data_validation_summary.json`
- `vectorization_comparison.csv`
- `model_diagnostics.csv`
- `topic_terms.csv`
- `topic_labels.csv`
- `representative_complaints.csv`
- `topic_prevalence.csv`
- `topic_prevalence_labeled.csv`
- `selected_model_summary.json`

Figures:

- `topic_count_npmi_coherence.png`
- `lda_topic_prevalence.png`
- `lsa_topic_prevalence.png`

## Interpretation caution

LDA topic weights are probabilities. LSA components are signed latent dimensions, not probability distributions. LSA document assignment in the output is therefore descriptive and is based on the largest absolute component loading.

Exact duplicate complaints are removed, but near-duplicate legal or dispute templates remain in the public sample and can influence the learned topics.
