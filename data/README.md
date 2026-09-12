# Data directory

The complaint records originate from the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**:

https://www.consumerfinance.gov/data-research/consumer-complaints/

For reproducibility, this project analyzes the fixed public 5,000-row `complaints_sample.csv` snapshot selected in Phase 1 rather than a changing live export. `src/01_acquire_validate.py` downloads that technical snapshot from:

```text
https://raw.githubusercontent.com/andygreen-1/Text_Analysis_Consumer_Complaints/main/Data/complaints_sample.csv
```

The exact snapshot used for final verification has SHA-256:

```text
0f19b276e8f1863419a0cdcba6bf9ca2924046f21cd8958f79258b8ecfd81a4a
```

The acquisition script verifies this checksum before analysis. If the local file does not match, the pipeline stops instead of silently analyzing a different dataset.

## Verified sample scope

- 5,000 rows and 18 columns
- one product category: `Credit reporting, credit repair services, or other personal consumer reports`
- one issue: `Incorrect information on your report`
- seven structured sub-issue categories
- date range: 2019-10-01 to 2020-09-29

Because the selected sample is narrow, the portfolio interprets **subthemes within incorrect credit-reporting complaints** and does not generalize the results to all CFPB complaints or all consumers.

## Files created during execution

Generated CSV data files are intentionally excluded from Git and are recreated from the verified source:

- `complaints_sample.csv` - downloaded source snapshot
- `complaints_modeling.csv` - non-empty, de-duplicated narratives
- `complaints_preprocessed.csv` - cleaned text used by vectorization/modeling

See `../REPRODUCIBILITY.md` for the verified environment and reproduced diagnostics.
