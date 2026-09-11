# Phase 2 implementation status

## Completed in code and executed

The project covers the required technical workflow for the development phase:

1. acquire/validate the selected complaint dataset
2. remove empty narratives and duplicate narratives
3. preprocess the complaint text
4. build a Bag-of-Words representation
5. build a TF-IDF representation
6. fit LDA models across several candidate topic counts
7. fit LSA/TruncatedSVD models across the same candidate counts
8. compare topic counts quantitatively
9. extract top terms
10. identify representative complaints
11. calculate document assignment / prevalence
12. generate result tables and visualizations

## Executed corpus facts

The selected file contained 5,000 complaint rows. After removing 683 exact duplicate narratives, 4,317 records remained. One record became empty after preprocessing, leaving 4,316 documents for modeling.

The implementation also established that the sample is limited to the CFPB product category "Credit reporting, credit repair services, or other personal consumer reports" and the issue "Incorrect information on your report". The analysis is therefore interpreted as recurring subthemes within this complaint category rather than a representative sample of all CFPB complaints.

## Model-selection diagnostics

The executed code reports:

- NPMI coherence for both LDA and LSA
- topic diversity for both methods
- LDA perplexity
- cumulative explained variance for LSA

Candidate counts 4–8 were tested. The highest NPMI coherence was obtained by the 8-topic LDA solution (0.2411) and the 4-component LSA solution (0.3421). Topic diversity was used as the first tie-breaker. The 6-topic LDA result was close in coherence, so the 8-topic selection should not be treated as an unambiguous optimum.

## Development issue and response

Validation showed that the selected public sample was much narrower than a generic consumer-complaint corpus: all records concern incorrect information on credit reports. Rather than overgeneralize the findings, the interpretation was restricted to recurring subthemes within this category. Exact duplicates were removed, while the remaining near-duplicate legal or dispute templates are documented as a limitation because they can influence topic structure.

## Important interpretation rule

LDA topic weights are probabilities and support direct topic-prevalence interpretation. LSA components are signed latent dimensions, not probabilities. The LSA document shares in the results are therefore descriptive assignments based on the largest absolute component loading and must not be described as probabilistic prevalence.

See `RESULTS_SUMMARY.md` and `outputs/tables/` for the executed numerical findings.
