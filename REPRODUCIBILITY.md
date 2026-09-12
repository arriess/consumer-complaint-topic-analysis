# Reproducibility verification

The final project was independently re-run and checked against the committed reference results before Phase 3 submission.

## Verified environment

- Python: **3.13.5** (`.python-version`)
- pandas: **2.2.3**
- NumPy: **2.3.5**
- scikit-learn: **1.8.0**
- matplotlib: **3.10.8**
- SciPy: **1.17.0**

`requirements-lock.txt` pins the complete direct/transitive package environment used for final verification. `requirements.txt` keeps bounded direct-dependency ranges for a compatible installation.

## Dataset identity and provenance

The records originate from the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**:

https://www.consumerfinance.gov/data-research/consumer-complaints/

The analysis uses a fixed 5,000-row public snapshot so that results can be reproduced against a stable input rather than a changing live export. The exact technical snapshot URL is documented in `data/README.md`.

Selected source file: `complaints_sample.csv`

SHA-256:

```text
0f19b276e8f1863419a0cdcba6bf9ca2924046f21cd8958f79258b8ecfd81a4a
```

`src/01_acquire_validate.py` checks this hash on every run and stops if the file does not match the verified snapshot.

## Verified dataset scope

- source shape: **5,000 rows x 18 columns**
- product categories in snapshot: **1**
- issues in snapshot: **1**
- structured sub-issue categories: **7**
- date range: **2019-10-01 to 2020-09-29**
- missing/empty narratives before cleaning: **0**
- exact duplicate narratives removed: **683**
- documents after de-duplication: **4,317**
- documents becoming empty after preprocessing: **1**
- final modeled documents: **4,316**

All records belong to the product category `Credit reporting, credit repair services, or other personal consumer reports` and the issue `Incorrect information on your report`. The portfolio therefore restricts conclusions to recurring subthemes within this category.

## Re-run verification

The independent verification reproduced the core project facts and diagnostics:

- vector matrices: **4,316 x 5,000**
- non-zero entries: **255,559**
- sparsity: **98.82%**
- selected LDA model: **8 topics**, NPMI **0.2411**, topic diversity **0.6375**, perplexity **759.25**
- selected LSA model: **4 components**, NPMI **0.3421**, topic diversity **0.8000**, cumulative explained variance **4.81%**
- largest LDA assignment share: **53.82%**

The 6-topic LDA solution reproduced an NPMI coherence of approximately **0.2390**, confirming that the 8-topic selection is only marginally stronger on fixed-seed coherence and should be interpreted cautiously.

## Multi-seed robustness check

A final robustness check evaluated the selected neighborhood across random seeds **7, 21, 42, 84, and 123**. This does not replace the submitted fixed-seed models; it tests sensitivity to random initialization.

- LDA k=6 mean NPMI: **0.2156**; mean topic diversity: **0.6767**; mean perplexity: **858.08**
- LDA k=8 mean NPMI: **0.2184**; mean topic diversity: **0.6375**; mean perplexity: **780.59**
- LSA k=4 reproduced NPMI **0.3421**, topic diversity **0.8000**, and explained variance **4.81%** for every tested seed

The stress test supports retaining the original 8-topic LDA as a defensible granular solution while making clear that LDA topic structure is seed-sensitive and should not be presented as uniquely optimal. Executed values are committed in `outputs/tables/stability_check.csv`.

## Run commands

Create/activate a Python 3.13.5 virtual environment, then install the verified environment and execute:

```bash
pip install -r requirements-lock.txt
python run_analysis.py
```

Optional robustness check:

```bash
python src/04_stability_check.py
```

The historical `run_phase2.py` runner is retained because it was used during development; both core runners execute the same three core analysis stages.

## Automated vs. interpreted outputs

The Python pipeline automatically generates validation statistics, vectorization comparison, model diagnostics, topic terms, representative complaints, assignment shares, selected-model metadata, and figures. `topic_labels.csv` and `topic_prevalence_labeled.csv` are human-interpreted presentation tables created after inspecting automated top terms, representative complaints, and assignment shares; they are not presented as algorithmically generated labels. `INTERPRETATION_NOTES.md` records the top-term evidence supporting each label.

Representative narrative exports are recreated locally but not committed to the public repository. This preserves reviewability through aggregate/term-level evidence while avoiding an unnecessary public duplicate of complaint narrative text.
