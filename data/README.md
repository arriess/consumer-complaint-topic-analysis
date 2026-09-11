# Data directory

`complaints_sample.csv` is downloaded by `src/01_acquire_validate.py` from the public dataset selected in Phase 1.

Selected source URL:

```text
https://raw.githubusercontent.com/andygreen-1/Text_Analysis_Consumer_Complaints/main/Data/complaints_sample.csv
```

The exact source snapshot used for the final verification has SHA-256:

```text
0f19b276e8f1863419a0cdcba6bf9ca2924046f21cd8958f79258b8ecfd81a4a
```

Generated data files are intentionally excluded from Git by default. This keeps the repository lightweight and allows the analysis to be rebuilt from the documented public source.

Files created during execution:

- `complaints_sample.csv` - downloaded source snapshot
- `complaints_modeling.csv` - non-empty, de-duplicated narratives
- `complaints_preprocessed.csv` - cleaned text used by vectorization/modeling

See `../REPRODUCIBILITY.md` for the verified environment and reproduced diagnostics.
