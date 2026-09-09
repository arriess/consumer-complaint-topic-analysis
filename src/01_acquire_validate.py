"""
Phase 2 - Step 1: Acquire and validate the CFPB complaint sample.

This script deliberately does NOT invent or silently resample data.
It downloads the exact public complaints_sample.csv selected in Phase 1,
checks its structure, reports data-quality facts, and prepares a modeling
dataset containing non-empty, unique complaint narratives.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "andygreen-1/Text_Analysis_Consumer_Complaints/"
    "main/Data/complaints_sample.csv"
)
RAW_PATH = DATA_DIR / "complaints_sample.csv"
MODELING_PATH = DATA_DIR / "complaints_modeling.csv"
SUMMARY_PATH = OUTPUT_DIR / "data_validation_summary.json"

EXPECTED_ROWS = 5000
EXPECTED_COLUMNS = 18

NARRATIVE_CANDIDATES = (
    "Consumer complaint narrative",
    "Consumer_complaint_narrative",
)


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns, candidates):
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def download_if_needed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 0:
        print(f"[OK] Dataset already exists: {RAW_PATH}")
        return

    print(f"[INFO] Downloading dataset from:\n{DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection, or manually "
            f"download complaints_sample.csv and place it at:\n{RAW_PATH}"
        ) from exc

    print(f"[OK] Downloaded: {RAW_PATH}")


def main() -> None:
    download_if_needed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("\n=== RAW DATASET ===")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if len(df) != EXPECTED_ROWS:
        print(
            f"[WARNING] Expected {EXPECTED_ROWS:,} rows from the selected snapshot, "
            f"but found {len(df):,}. No automatic resampling was performed."
        )
    else:
        print(f"[OK] Expected row count confirmed: {EXPECTED_ROWS:,}")

    if len(df.columns) != EXPECTED_COLUMNS:
        print(
            f"[WARNING] Expected {EXPECTED_COLUMNS} columns, but found "
            f"{len(df.columns)}. Continue only after checking the source."
        )
    else:
        print(f"[OK] Expected column count confirmed: {EXPECTED_COLUMNS}")

    narrative_col = find_column(df.columns, NARRATIVE_CANDIDATES)
    if narrative_col is None:
        raise KeyError(
            "Could not find the complaint narrative column. Available columns:\n"
            + "\n".join(map(str, df.columns))
        )

    print(f"[OK] Narrative column: {narrative_col!r}")

    raw_narratives = df[narrative_col]
    nonempty_mask = raw_narratives.notna() & raw_narratives.astype(str).str.strip().ne("")
    nonempty = df.loc[nonempty_mask].copy()

    missing_or_empty = int((~nonempty_mask).sum())
    duplicate_narratives = int(
        nonempty[narrative_col].astype(str).str.strip().duplicated().sum()
    )

    # Preserve first occurrence only, exactly as declared in Phase 1.
    nonempty[narrative_col] = nonempty[narrative_col].astype(str).str.strip()
    modeling = nonempty.drop_duplicates(subset=[narrative_col], keep="first").copy()

    word_counts = modeling[narrative_col].str.split().str.len()

    # Add a stable normalized field while retaining all original metadata.
    if narrative_col != "consumer_complaint_narrative":
        modeling["consumer_complaint_narrative"] = modeling[narrative_col]

    modeling.to_csv(MODELING_PATH, index=False)

    summary = {
        "source_url": DATA_URL,
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
        "expected_rows": EXPECTED_ROWS,
        "expected_columns": EXPECTED_COLUMNS,
        "narrative_column_detected": narrative_col,
        "missing_or_empty_narratives": missing_or_empty,
        "duplicate_nonempty_narratives": duplicate_narratives,
        "modeling_rows_after_nonempty_and_deduplication": int(len(modeling)),
        "document_word_count": {
            "min": int(word_counts.min()) if len(word_counts) else None,
            "median": float(word_counts.median()) if len(word_counts) else None,
            "mean": float(word_counts.mean()) if len(word_counts) else None,
            "max": int(word_counts.max()) if len(word_counts) else None,
        },
    }

    for col in ("Product", "Issue"):
        if col in modeling.columns:
            summary[f"top_{col.lower()}_counts"] = {
                str(k): int(v)
                for k, v in modeling[col].fillna("<missing>").value_counts().head(10).items()
            }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DATA-QUALITY RESULTS ===")
    print(f"Missing/empty narratives:      {missing_or_empty:,}")
    print(f"Duplicate non-empty narratives:{duplicate_narratives:,}")
    print(f"Rows kept for modeling:        {len(modeling):,}")

    if len(word_counts):
        print(
            "Document length (words): "
            f"min={int(word_counts.min())}, "
            f"median={word_counts.median():.1f}, "
            f"mean={word_counts.mean():.1f}, "
            f"max={int(word_counts.max())}"
        )

    print(f"\n[OK] Modeling dataset: {MODELING_PATH}")
    print(f"[OK] Validation summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
