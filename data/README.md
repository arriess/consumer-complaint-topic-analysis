# Data directory

`complaints_sample.csv` is downloaded by `src/01_acquire_validate.py` from the
public dataset selected in Phase 1.

Generated files are intentionally excluded from Git by default. This keeps the
repository lightweight and makes the analysis reproducible from the source URL.

Files created during execution:

- `complaints_sample.csv` — downloaded source snapshot
- `complaints_modeling.csv` — non-empty, de-duplicated narratives
- `complaints_preprocessed.csv` — cleaned text used by vectorization/modeling
