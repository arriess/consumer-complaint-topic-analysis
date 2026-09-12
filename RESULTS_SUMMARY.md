# Executed Results Summary

## Dataset scope discovered during implementation

The selected `complaints_sample.csv` contains exactly **5,000 rows and 18 columns**. The complaint records originate from the CFPB Consumer Complaint Database; the project uses a fixed public snapshot for reproducibility.

Every record belongs to the product category **Credit reporting, credit repair services, or other personal consumer reports** and the issue **Incorrect information on your report**. The snapshot nevertheless contains **7 structured sub-issue categories**, so there is meaningful variation within the fixed issue. The date range is **2019-10-01 to 2020-09-29**.

The implemented analysis is therefore described as recurring **subthemes within incorrect credit-reporting complaints**, not as a representative analysis of all CFPB complaints or all consumers.

## Data cleaning

- Raw complaints: **5,000**
- Missing/empty narratives before cleaning: **0**
- Exact duplicate narratives removed: **683**
- Narratives after duplicate removal: **4,317**
- Empty narratives after text preprocessing: **1**
- Final documents modeled: **4,316**
- Median raw document length before preprocessing: **95 words**
- Mean raw document length before preprocessing: **158.6 words**
- Median cleaned length: **34 tokens**
- Mean cleaned length: **61.9 tokens**

The executed preprocessing follows the submitted Phase 1 plan: lowercase text; remove URLs/e-mail artefacts, CFPB redaction placeholders, punctuation and numeric noise; tokenize with a deterministic regex; remove scikit-learn English stop words; and remove one-character tokens. No lemmatization is claimed or performed.

## Vectorization comparison

Both representations use the same 5,000-feature unigram/bigram vocabulary (`min_df=5`, `max_df=0.95`, `max_features=5000`). TF-IDF additionally uses `sublinear_tf=True`.

| Method | Documents | Features | Non-zero entries | Sparsity | Matrix values |
|---|---:|---:|---:|---:|---|
| Bag of Words | 4,316 | 5,000 | 255,559 | 98.82% | raw term counts |
| TF-IDF | 4,316 | 5,000 | 255,559 | 98.82% | TF-IDF weights |

The sparsity pattern is identical because both methods use the same documents and vocabulary. Their information weighting differs: Bag of Words preserves raw counts, whereas TF-IDF reduces the influence of terms occurring across much of the corpus.

## Topic-count diagnostics

Candidate counts **4, 5, 6, 7, and 8** were tested. The code selects the highest NPMI coherence within each method, with topic diversity as the first tie-breaker. Perplexity and explained variance are retained as method-specific diagnostics rather than mixed into a single artificial score.

### LDA

The highest fixed-seed NPMI coherence among the tested LDA models occurred at **8 topics**:

- NPMI coherence: **0.2411**
- topic diversity: **0.6375**
- perplexity: **759.25**

The 6-topic solution was very close in coherence (**0.2390**), so the final interpretation does not treat 8 topics as an unambiguous optimum.

### LSA

The highest NPMI coherence among the tested LSA solutions occurred at **4 components**:

- NPMI coherence: **0.3421**
- topic diversity: **0.8000**
- cumulative explained variance: **4.81%**

Explained variance increases as components are added, so it is treated as a descriptive diagnostic rather than the sole selection criterion.

## Interpreted LDA topics

1. Credit-file inaccuracies and mixed disputes
2. Account opening, balance, and charge details
3. Identity theft and fraudulent accounts
4. Debt collection and legal-compliance allegations
5. Formal inaccurate-information correction requests
6. Payment, loan, and account-status reporting
7. Identity-theft blocking and FCRA rights
8. Furnisher and inquiry evidence disputes

The dominant LDA topic was Topic 6, assigned to **53.82%** of modeled documents. Topic 3 accounted for **19.11%**, and Topic 1 for **16.52%**. Several remaining topics were small, reinforcing the need to interpret the 8-topic solution cautiously.

## Interpreted LSA components

1. General credit-report and account disputes
2. Fraudulent charge and account-opening details
3. Unknown or fraudulent credit-file items
4. Identity-theft victim and blocking-rights language

LSA component assignments use the largest absolute component loading. They are **not probabilities** and are therefore described as document assignments rather than probabilistic prevalence.

`INTERPRETATION_NOTES.md` documents the top-term evidence supporting every human-assigned topic/component label.

## Robustness check

A post-finalization stress test repeated LDA k=6, LDA k=8, and LSA k=4 across random seeds **7, 21, 42, 84, and 123**:

- LDA k=6: mean NPMI **0.2156**, mean diversity **0.6767**, mean perplexity **858.08**
- LDA k=8: mean NPMI **0.2184**, mean diversity **0.6375**, mean perplexity **780.59**
- LSA k=4: NPMI **0.3421**, diversity **0.8000**, explained variance **4.81%** for every tested seed

This supports retaining the submitted LDA-8 solution as a defensible granular view while making the seed sensitivity explicit instead of implying unique optimality.

## Method comparison and limitations

LDA produced a more granular set of complaint themes and directly supplies document-topic probabilities, which supports probabilistic assignment interpretation. LSA produced fewer, broader latent semantic dimensions and higher selected-model NPMI coherence, but its signed components are mathematical directions rather than probability distributions.

The corpus also contains repeated legal and dispute-template language. Exact duplicates were removed, but near-duplicate templates remain and can influence both models. The fixed sample is narrow and the CFPB database is not a statistical sample of all consumer experiences, so conclusions are intentionally limited to the selected complaint subset.
