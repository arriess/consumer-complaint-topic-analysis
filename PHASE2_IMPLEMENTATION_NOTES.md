# Phase 2 implementation status

## Completed in code

The project now covers the required technical workflow for the development
phase:

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

## Model-selection diagnostics

The code reports:

- NPMI coherence for both LDA and LSA
- topic diversity for both methods
- LDA perplexity
- cumulative explained variance for LSA

The selected topic count is the candidate with the highest executed NPMI
coherence, using topic diversity as the first tie-breaker.

This rule is deliberately explicit so the final written portfolio can explain
why a particular topic count was selected.

## Important interpretation rule

LDA topic weights are probabilities. They can be interpreted as a document's
topic mixture.

LSA components are not probabilities and can have positive or negative values.
The code therefore uses absolute component magnitude only for identifying
strongly associated representative documents and for a descriptive component
assignment. The final report should not describe LSA assignment shares as
probabilistic topic prevalence.

## Still dependent on execution

Do not write final numerical findings until the pipeline has actually run.

The following must come from the generated output files:

- number of usable complaint narratives
- vocabulary size
- vector-matrix sparsity
- selected LDA topic count
- selected LSA component count
- coherence values
- perplexity
- explained variance
- actual top terms
- topic labels
- representative complaint examples
- prevalence/assignment shares
- final interpretation and limitations

Once those outputs exist, the Phase 2 written reflection can be drafted using
only observed results.
