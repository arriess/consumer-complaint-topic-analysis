# Reproducibility verification

The final project was independently re-run and checked against the committed reference results before Phase 3 submission.

## Verified environment

- Python: **3.13.5**
- pandas: **2.2.3**
- NumPy: **2.3.5**
- scikit-learn: **1.8.0**
- matplotlib: **3.10.8**
- SciPy: **1.17.0**

These exact package versions are recorded in `requirements-lock.txt`. `requirements.txt` keeps compatible version ranges for normal installation.

## Dataset identity

Selected source file: `complaints_sample.csv`

SHA-256:

```text
0f19b276e8f1863419a0cdcba6bf9ca2924046f21cd8958f79258b8ecfd81a4a
```

## Re-run verification

The verification run reproduced the core project facts and diagnostics:

- source shape: **5,000 rows x 18 columns**
- missing/empty narratives before cleaning: **0**
- exact duplicate narratives removed: **683**
- documents after de-duplication: **4,317**
- documents becoming empty after preprocessing: **1**
- final modeled documents: **4,316**
- vector matrices: **4,316 x 5,000**
- non-zero entries: **255,559**
- sparsity: **98.82%**
- selected LDA model: **8 topics**, NPMI **0.2411**, topic diversity **0.6375**, perplexity **759.25**
- selected LSA model: **4 components**, NPMI **0.3421**, topic diversity **0.8000**, cumulative explained variance **4.81%**
- largest LDA assignment share: **53.82%**

The 6-topic LDA solution reproduced an NPMI coherence of approximately **0.2390**, confirming that the 8-topic selection is only marginally stronger on the fixed-seed coherence result and should be interpreted cautiously.

## Multi-seed robustness check

A final robustness check evaluated the selected neighborhood across random seeds **7, 21, 42, 84, and 123**. This does not replace the submitted fixed-seed models; it tests how sensitive the conclusion is to random initialization.

- 6-topic LDA mean NPMI: **0.2156**; mean perplexity: **858.08**
- 8-topic LDA mean NPMI: **0.2184**; mean perplexity: **780.59**
- 8-topic LDA therefore retained a small mean coherence advantage and substantially lower mean perplexity, but its topic diversity was lower (**0.6375** mean vs. **0.6767** for k=6).
- 4-component LSA reproduced NPMI **0.3421**, topic diversity **0.8000**, and explained variance **4.81%** for every tested seed.

The stress test supports retaining the original 8-topic LDA as a defensible granular solution while making clear that LDA topic structure is seed-sensitive and should not be presented as uniquely optimal. The executed values are committed in `outputs/tables/stability_check.csv`; the optional check can be reproduced with `python src/04_stability_check.py` after the core pipeline.

## Run commands

For the final project use:

```bash
python run_analysis.py
```

Optional multi-seed robustness check:

```bash
python src/04_stability_check.py
```

The historical `run_phase2.py` runner is retained because it was used during the development phase; both core runners execute the same three analysis stages.

## Automated vs interpreted outputs

The Python pipeline automatically generates validation statistics, vectorization comparison, model diagnostics, topic terms, representative complaints, assignment shares, selected-model metadata, and figures. `topic_labels.csv` and `topic_prevalence_labeled.csv` are human-interpreted presentation tables created after inspecting the automated top terms, representative complaints, and assignment shares; they are not presented as algorithmically generated labels.
