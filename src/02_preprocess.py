"""
Phase 2 - Step 2: Clean complaint narratives for topic modeling.

Deterministic pipeline:
- lowercase
- remove URLs and e-mail artefacts
- remove CFPB-style redaction placeholders
- remove punctuation and numeric noise
- regex tokenization
- remove English stop words
- save clean text for vectorization

This implementation follows the submitted Phase 1 preprocessing plan and uses
scikit-learn's built-in English stop-word list with deterministic regex
tokenization so the workflow can be reproduced without external language-model
downloads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "complaints_modeling.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "complaints_preprocessed.csv"

TEXT_COLUMN = "consumer_complaint_narrative"
CLEAN_COLUMN = "clean_text"

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# CFPB narratives commonly mask information with XX / XXXX patterns.
REDACTION_RE = re.compile(r"\b(?:x{2,}|xx(?:/xx)+(?:/xxxx)?)\b", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = REDACTION_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    tokens = [
        token
        for token in re.findall(r"[a-z]+", text)
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. Run 01_acquire_validate.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    if TEXT_COLUMN not in df.columns:
        raise KeyError(
            f"Expected normalized narrative column {TEXT_COLUMN!r}. "
            "Run 01_acquire_validate.py first."
        )

    before = len(df)
    df[CLEAN_COLUMN] = df[TEXT_COLUMN].fillna("").map(clean_text)

    empty_after_cleaning = int(df[CLEAN_COLUMN].str.strip().eq("").sum())
    df = df.loc[df[CLEAN_COLUMN].str.strip().ne("")].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    token_counts = df[CLEAN_COLUMN].str.split().str.len()

    print("=== PREPROCESSING RESULTS ===")
    print(f"Input rows:                {before:,}")
    print(f"Empty after preprocessing: {empty_after_cleaning:,}")
    print(f"Output rows:               {len(df):,}")
    if len(token_counts):
        print(
            "Clean tokens/document: "
            f"median={token_counts.median():.1f}, "
            f"mean={token_counts.mean():.1f}"
        )
    print("[NOTE] Deterministic preprocessing: sklearn stop words and regex tokenization.")
    print(f"[OK] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
