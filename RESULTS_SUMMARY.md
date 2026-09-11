# Executed Phase 2 Results

## Scope correction discovered during implementation

The uploaded `complaints_sample.csv` contains exactly **5,000 rows and 18 columns**.
Every record belongs to the product category **"Credit reporting, credit repair services, or other personal consumer reports"** and the issue **"Incorrect information on your report"**. Therefore, the implemented analysis is described as recurring **subthemes within incorrect credit-reporting complaints**, not as a representative analysis of all CFPB consumer complaints.

The date range in the sample is **2019-10-01 to 2020-09-29**.

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

The executed preprocessing follows the submitted Phase 1 plan: lowercase text; remove URLs/e-mail artefacts, CFPB redaction placeholders, punctuation and numeric noise; tokenize with a deterministic regex; and remove scikit-learn English stop words.

## Vectorization

Both representations use the same 5,000-feature unigram/bigram vocabulary (`min_df=5`, `max_df=0.95`).

| Method | Documents | Features | Non-zero entries | Sparsity |
|---|---:|---:|---:|---:|
| Bag of Words | 4,316 | 5,000 | 255,559 | 98.82% |
| TF-IDF | 4,316 | 5,000 | 255,559 | 98.82% |

The sparsity pattern is the same because both methods use the same documents and vocabulary. The difference is the matrix values: raw term counts for Bag of Words versus reweighted TF-IDF values.

## Topic-count diagnostics

Candidate counts **4, 5, 6, 7, and 8** were tested.

### LDA

The highest NPMI coherence among the tested LDA models occurred at **8 topics**:

- NPMI coherence: **0.2411**
- topic diversity: **0.6375**
- perplexity: **759.25**

The 6-topic solution was very close in coherence (**0.2390**), so the final interpretation should acknowledge that 8 is not an unambiguous optimum.

### LSA

The highest NPMI coherence among the tested LSA solutions occurred at **4 components**:

- NPMI coherence: **0.3421**
- topic diversity: **0.8000**
- cumulative explained variance: **4.81%**

Explained variance increases as more components are added, so it was treated as a descriptive diagnostic rather than the sole selection criterion.

## Interpreted LDA topics

1. Credit-file inaccuracies and mixed disputes
2. Account opening, balance, and charge details
3. Identity theft and fraudulent accounts
4. Debt collection and legal-compliance allegations
5. Formal inaccurate-information correction requests
6. Payment, loan, and account-status reporting
7. Identity-theft blocking and FCRA rights
8. Furnisher and inquiry evidence disputes

The dominant LDA topic was Topic 6, assigned to **53.82%** of modeled documents. Topic 3 accounted for **19.11%**, and Topic 1 for **16.52%**. Several other topics were small, which is another reason to treat the 8-topic solution cautiously.

## Interpreted LSA components

1. General credit-report and account disputes
2. Fraudulent charge and account-opening details
3. Unknown or fraudulent credit-file items
4. Identity-theft victim and blocking-rights language

LSA component assignments are based on the largest absolute component loading. They are **not probabilities** and should not be described as probabilistic topic prevalence.

## Method comparison

LDA produced a more granular set of complaint themes and directly supplies document-topic probabilities, which makes prevalence analysis easier. LSA produced fewer, broader latent dimensions and achieved higher NPMI coherence for its selected 4-component solution, but its components are signed mathematical directions rather than probability distributions.

The corpus also contains repeated legal and dispute-template language. Exact duplicates were removed, but near-duplicate templates remain and can influence both models. This is an important limitation for the final portfolio reflection.
