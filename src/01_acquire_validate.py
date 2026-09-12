"""
Phase 2 - Step 1: Acquire and validate the selected CFPB complaint sample.

The script downloads the exact public complaints_sample.csv snapshot selected in
Phase 1, verifies its SHA-256 identity, checks the expected structure, reports
data-quality and scope facts, and prepares a modeling dataset containing
non-empty, unique complaint narratives.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

# The original records come from the CFPB Consumer Complaint Database. The
# project uses the fixed public 5,000-row snapshot below so the portfolio can be
# reproduced against an immutable, verifiable input rather than a changing live
# database export.
OFFICIAL_CFPB_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/"
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
EXPECTED_SHA256 = "0f19b276e8f1863419a0cdcba6bf9ca2924046f21cd8958f79258b8ecfd81a4a"

NARRATIVE_CANDIDATES = (
    "Consumer complaint narrative",
    "Consumer_complaint_narrative",
    "consumer_complaint_narrative",
)


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(columns, candidates):
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_identity(path: Path) -> str:
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            "Dataset checksum mismatch. The file is not the verified snapshot "
            f"used for this project. Expected {EXPECTED_SHA256}, found {actual}."
        )
    print(f"[OK] Dataset SHA-256 confirmed: {actual}")
    return actual


def download_if_needed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 0:
        print(f"[OK] Dataset already exists: {RAW_PATH}")
        verify_source_identity(RAW_PATH)
        return

    print(f"[INFO] Downloading verified snapshot from:\n{DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Dataset download failed. Check the internet connection, or manually "
            f"place the verified complaints_sample.csv at:\n{RAW_PATH}"
        ) from exc

    verify_source_identity(RAW_PATH)
    print(f"[OK] Downloaded: {RAW_PATH}")


def counts_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("<missing>").value_counts(dropna=False).items()
    }


def main() -> None:
    download_if_needed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_sha256 = verify_source_identity(RAW_PATH)
    df = pd.read_csv(RAW_PATH)

    print("\n=== RAW DATASET ===")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows from the verified snapshot, "
            f"but found {len(df):,}."
        )
    print(f"[OK] Expected row count confirmed: {EXPECTED_ROWS:,}")

    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns from the verified snapshot, "
            f"but found {len(df.columns)}."
        )
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

    nonempty[narrative_col] = nonempty[narrative_col].astype(str).str.strip()
    modeling = nonempty.drop_duplicates(subset=[narrative_col], keep="first").copy()
    word_counts = modeling[narrative_col].str.split().str.len()

    # Add a stable normalized field while retaining all original metadata.
    if narrative_col != "consumer_complaint_narrative":
        modeling["consumer_complaint_narrative"] = modeling[narrative_col]

    modeling.to_csv(MODELING_PATH, index=False)

    summary = {
        "official_source": OFFICIAL_CFPB_URL,
        "technical_snapshot_url": DATA_URL,
        "source_sha256": source_sha256,
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

    product_col = find_column(df.columns, ("product",))
    issue_col = find_column(df.columns, ("issue",))
    sub_issue_col = find_column(df.columns, ("sub_issue", "sub issue"))
    date_col = find_column(df.columns, ("date_received", "date received"))

    if product_col:
        summary["distinct_product_count"] = int(df[product_col].nunique(dropna=False))
        summary["product_counts"] = counts_dict(df[product_col])
    if issue_col:
        summary["distinct_issue_count"] = int(df[issue_col].nunique(dropna=False))
        summary["issue_counts"] = counts_dict(df[issue_col])
    if sub_issue_col:
        summary["distinct_sub_issue_count"] = int(df[sub_issue_col].nunique(dropna=False))
        summary["sub_issue_counts"] = counts_dict(df[sub_issue_col])
    if date_col:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
        summary["date_received_min"] = (
            parsed_dates.min().date().isoformat() if parsed_dates.notna().any() else None
        )
        summary["date_received_max"] = (
            parsed_dates.max().date().isoformat() if parsed_dates.notna().any() else None
        )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DATA-QUALITY RESULTS ===")
    print(f"Missing/empty narratives:       {missing_or_empty:,}")
    print(f"Duplicate non-empty narratives: {duplicate_narratives:,}")
    print(f"Rows kept for modeling:         {len(modeling):,}")
    if product_col:
        print(f"Distinct products:              {summary['distinct_product_count']:,}")
    if issue_col:
        print(f"Distinct issues:                {summary['distinct_issue_count']:,}")
    if sub_issue_col:
        print(f"Distinct sub-issues:            {summary['distinct_sub_issue_count']:,}")
    if date_col:
        print(
            "Date range:                       "
            f"{summary['date_received_min']} to {summary['date_received_max']}"
        )

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
