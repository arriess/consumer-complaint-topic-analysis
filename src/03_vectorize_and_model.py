"""
Phase 2 - Step 3: Vectorization, topic modeling, diagnostics, representative
documents, prevalence, and visualizations.

This script implements the Phase 1 plan without inventing results:
- Bag of Words with CountVectorizer
- TF-IDF with TfidfVectorizer
- LDA on Bag of Words
- LSA with TruncatedSVD on TF-IDF
- candidate topic-count comparison
- NPMI topic coherence and topic diversity
- method-specific diagnostics
- representative complaint extraction
- topic/component prevalence
- output tables and figures

Run this only after:
    python src/01_acquire_validate.py
    python src/02_preprocess.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "complaints_preprocessed.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TEXT_COLUMN = "clean_text"
RAW_TEXT_COLUMN = "consumer_complaint_narrative"

RANDOM_STATE = 42

# Candidate topic counts to evaluate. The final choice is based on executed
# diagnostics, not assumed in advance.
TOPIC_COUNTS = [4, 5, 6, 7, 8]

TOP_WORDS = 10
REPRESENTATIVE_DOCS_PER_TOPIC = 3

# These values are transparent modeling choices and can be revised after
# inspecting the actual corpus diagnostics.
MIN_DF = 5
MAX_DF = 0.95
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def matrix_sparsity(matrix) -> float:
    total = matrix.shape[0] * matrix.shape[1]
    if total == 0:
        return float("nan")
    nonzero = matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix)
    return 1.0 - (nonzero / total)


def get_top_term_indices(components: np.ndarray, n_top_words: int) -> list[list[int]]:
    """
    Return indices of the highest-weight positive terms for each component.
    For LSA, SVD component signs are mathematically arbitrary; using the
    positive direction gives one consistent, reproducible orientation.
    """
    return [
        np.argsort(component)[::-1][:n_top_words].tolist()
        for component in components
    ]


def topic_diversity(top_indices: list[list[int]]) -> float:
    if not top_indices:
        return float("nan")
    flattened = [idx for topic in top_indices for idx in topic]
    if not flattened:
        return float("nan")
    return len(set(flattened)) / len(flattened)


def build_binary_matrix(count_matrix):
    binary = count_matrix.copy().tocsr()
    binary.data = np.ones_like(binary.data)
    return binary


def npmi_coherence(
    binary_doc_term,
    top_indices: list[list[int]],
    eps: float = 1e-12,
) -> float:
    """Efficient mean NPMI across pairs of top terms within each topic."""
    n_docs = binary_doc_term.shape[0]
    if n_docs == 0:
        return float("nan")

    unique_terms = sorted({idx for topic in top_indices for idx in topic})
    if not unique_terms:
        return float("nan")

    local_pos = {term_idx: pos for pos, term_idx in enumerate(unique_terms)}
    sub = binary_doc_term[:, unique_terms].astype(np.float64)
    doc_freq = np.asarray(sub.sum(axis=0)).ravel()
    cooc = (sub.T @ sub).toarray()

    topic_scores = []
    for topic in top_indices:
        pair_scores = []
        for i in range(len(topic)):
            li = local_pos[topic[i]]
            p_i = max(float(doc_freq[li]) / n_docs, eps)
            for j in range(i + 1, len(topic)):
                lj = local_pos[topic[j]]
                p_j = max(float(doc_freq[lj]) / n_docs, eps)
                p_ij = max(float(cooc[li, lj]) / n_docs, eps)
                pmi = math.log(p_ij / (p_i * p_j))
                denom = -math.log(p_ij)
                if denom > 0:
                    pair_scores.append(pmi / denom)
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(topic_scores)) if topic_scores else float("nan")


def extract_topics(
    method: str,
    components: np.ndarray,
    feature_names: np.ndarray,
    n_top_words: int = TOP_WORDS,
) -> pd.DataFrame:
    rows = []
    top_indices = get_top_term_indices(components, n_top_words)

    for topic_idx, indices in enumerate(top_indices, start=1):
        terms = [str(feature_names[i]) for i in indices]
        weights = [float(components[topic_idx - 1, i]) for i in indices]

        for rank, (term, weight) in enumerate(zip(terms, weights), start=1):
            rows.append(
                {
                    "method": method,
                    "topic_id": topic_idx,
                    "term_rank": rank,
                    "term": term,
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows)


def evaluate_lda(
    X_bow,
    binary_bow,
    feature_names,
    topic_counts,
) -> tuple[pd.DataFrame, dict[int, LatentDirichletAllocation]]:
    rows = []
    models = {}

    for k in topic_counts:
        print(f"[INFO] Fitting LDA with k={k}")
        model = LatentDirichletAllocation(
            n_components=k,
            learning_method="online",
            max_iter=5,
            batch_size=256,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_bow)
        models[k] = model

        top_indices = get_top_term_indices(model.components_, TOP_WORDS)

        rows.append(
            {
                "method": "LDA",
                "n_topics": k,
                "npmi_coherence": npmi_coherence(binary_bow, top_indices),
                "topic_diversity": topic_diversity(top_indices),
                "perplexity": float(model.perplexity(X_bow)),
                "explained_variance": np.nan,
            }
        )

    return pd.DataFrame(rows), models


def evaluate_lsa(
    X_tfidf,
    binary_bow,
    feature_names,
    topic_counts,
) -> tuple[pd.DataFrame, dict[int, TruncatedSVD]]:
    rows = []
    models = {}

    for k in topic_counts:
        print(f"[INFO] Fitting LSA with k={k}")
        model = TruncatedSVD(
            n_components=k,
            algorithm="randomized",
            n_iter=10,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tfidf)
        models[k] = model

        top_indices = get_top_term_indices(model.components_, TOP_WORDS)

        rows.append(
            {
                "method": "LSA",
                "n_topics": k,
                "npmi_coherence": npmi_coherence(binary_bow, top_indices),
                "topic_diversity": topic_diversity(top_indices),
                "perplexity": np.nan,
                "explained_variance": float(
                    np.sum(model.explained_variance_ratio_)
                ),
            }
        )

    return pd.DataFrame(rows), models


def select_topic_count(diagnostics: pd.DataFrame, method: str) -> int:
    """
    Select the candidate with highest NPMI coherence.
    Topic diversity is the first tie-breaker.

    Method-specific metrics are still preserved in the output table:
    - LDA perplexity
    - LSA cumulative explained variance

    This makes the selection transparent while retaining comparable and
    method-specific diagnostics for interpretation.
    """
    subset = diagnostics.loc[diagnostics["method"] == method].copy()

    if subset["npmi_coherence"].notna().any():
        subset = subset.sort_values(
            ["npmi_coherence", "topic_diversity", "n_topics"],
            ascending=[False, False, True],
        )
        return int(subset.iloc[0]["n_topics"])

    # Fallback only if coherence could not be computed.
    if method == "LDA":
        subset = subset.sort_values(
            ["perplexity", "topic_diversity", "n_topics"],
            ascending=[True, False, True],
        )
    else:
        subset = subset.sort_values(
            ["explained_variance", "topic_diversity", "n_topics"],
            ascending=[False, False, True],
        )
    return int(subset.iloc[0]["n_topics"])


def representative_documents(
    df: pd.DataFrame,
    method: str,
    document_scores: np.ndarray,
    n_per_topic: int = REPRESENTATIVE_DOCS_PER_TOPIC,
) -> pd.DataFrame:
    rows = []

    if method == "LDA":
        assignment_scores = document_scores
    else:
        # LSA coordinates are signed and component sign is arbitrary.
        # Absolute magnitude is used only to identify documents most strongly
        # associated with each component.
        assignment_scores = np.abs(document_scores)

    for topic_idx in range(assignment_scores.shape[1]):
        scores = assignment_scores[:, topic_idx]
        best_indices = np.argsort(scores)[::-1][:n_per_topic]

        for rank, doc_idx in enumerate(best_indices, start=1):
            row = {
                "method": method,
                "topic_id": topic_idx + 1,
                "representative_rank": rank,
                "document_index": int(doc_idx),
                "topic_score": float(scores[doc_idx]),
                "clean_text": str(df.iloc[doc_idx][TEXT_COLUMN]),
            }
            if RAW_TEXT_COLUMN in df.columns:
                row["original_narrative"] = str(
                    df.iloc[doc_idx][RAW_TEXT_COLUMN]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def topic_prevalence(
    method: str,
    document_scores: np.ndarray,
) -> pd.DataFrame:
    if method == "LDA":
        assignment_scores = document_scores
        assignment_basis = "highest LDA topic probability"
    else:
        assignment_scores = np.abs(document_scores)
        assignment_basis = "largest absolute LSA component loading"

    assignments = np.argmax(assignment_scores, axis=1)
    counts = pd.Series(assignments).value_counts().sort_index()

    rows = []
    total = len(assignments)
    for topic_idx in range(document_scores.shape[1]):
        count = int(counts.get(topic_idx, 0))
        rows.append(
            {
                "method": method,
                "topic_id": topic_idx + 1,
                "document_count": count,
                "share": count / total if total else np.nan,
                "assignment_basis": assignment_basis,
            }
        )

    return pd.DataFrame(rows)


def save_diagnostic_figure(diagnostics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in diagnostics["method"].unique():
        subset = diagnostics.loc[diagnostics["method"] == method]
        ax.plot(
            subset["n_topics"],
            subset["npmi_coherence"],
            marker="o",
            label=method,
        )

    ax.set_title("Topic-count comparison: NPMI coherence")
    ax.set_xlabel("Number of topics/components")
    ax.set_ylabel("Mean NPMI coherence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "topic_count_npmi_coherence.png", dpi=180)
    plt.close(fig)


def save_prevalence_figure(prevalence: pd.DataFrame) -> None:
    methods = list(prevalence["method"].unique())
    for method in methods:
        subset = prevalence.loc[prevalence["method"] == method].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            subset["topic_id"].astype(str),
            subset["share"],
        )
        ax.set_title(f"{method}: document assignment by topic/component")
        ax.set_xlabel("Topic / component")
        ax.set_ylabel("Share of documents")
        ax.set_ylim(0, max(0.05, float(subset["share"].max()) * 1.15))
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"{method.lower()}_topic_prevalence.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> None:
    ensure_dirs()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. Run 01_acquire_validate.py and "
            "02_preprocess.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    if TEXT_COLUMN not in df.columns:
        raise KeyError(f"Missing required column: {TEXT_COLUMN}")

    texts = df[TEXT_COLUMN].fillna("").astype(str)
    nonempty_mask = texts.str.strip().ne("")
    df = df.loc[nonempty_mask].reset_index(drop=True)
    texts = df[TEXT_COLUMN].astype(str)

    if len(df) < 20:
        raise ValueError(
            "Too few usable documents for the planned topic modeling workflow."
        )

    print(f"[OK] Modeling documents: {len(df):,}")

    count_vectorizer = CountVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    tfidf_vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
    )

    X_bow = count_vectorizer.fit_transform(texts)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    bow_terms = count_vectorizer.get_feature_names_out()
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()

    if list(bow_terms) != list(tfidf_terms):
        raise RuntimeError(
            "BoW and TF-IDF vocabularies differ unexpectedly. "
            "Check vectorizer parameters."
        )

    vectorization_comparison = pd.DataFrame(
        [
            {
                "method": "Bag of Words",
                "documents": X_bow.shape[0],
                "features": X_bow.shape[1],
                "nonzero_entries": int(X_bow.nnz),
                "sparsity": matrix_sparsity(X_bow),
                "matrix_value_interpretation": "term counts",
            },
            {
                "method": "TF-IDF",
                "documents": X_tfidf.shape[0],
                "features": X_tfidf.shape[1],
                "nonzero_entries": int(X_tfidf.nnz),
                "sparsity": matrix_sparsity(X_tfidf),
                "matrix_value_interpretation": "TF-IDF weights",
            },
        ]
    )
    vectorization_comparison.to_csv(
        TABLE_DIR / "vectorization_comparison.csv",
        index=False,
    )

    binary_bow = build_binary_matrix(X_bow)

    lda_diag, lda_models = evaluate_lda(
        X_bow,
        binary_bow,
        bow_terms,
        TOPIC_COUNTS,
    )
    lsa_diag, lsa_models = evaluate_lsa(
        X_tfidf,
        binary_bow,
        tfidf_terms,
        TOPIC_COUNTS,
    )

    diagnostics = pd.concat([lda_diag, lsa_diag], ignore_index=True)
    diagnostics.to_csv(
        TABLE_DIR / "model_diagnostics.csv",
        index=False,
    )

    selected_lda_k = select_topic_count(diagnostics, "LDA")
    selected_lsa_k = select_topic_count(diagnostics, "LSA")

    lda_model = lda_models[selected_lda_k]
    lsa_model = lsa_models[selected_lsa_k]

    lda_doc_scores = lda_model.transform(X_bow)
    lsa_doc_scores = lsa_model.transform(X_tfidf)

    lda_topics = extract_topics(
        "LDA",
        lda_model.components_,
        bow_terms,
    )
    lsa_topics = extract_topics(
        "LSA",
        lsa_model.components_,
        tfidf_terms,
    )
    topics = pd.concat([lda_topics, lsa_topics], ignore_index=True)
    topics.to_csv(TABLE_DIR / "topic_terms.csv", index=False)

    reps = pd.concat(
        [
            representative_documents(df, "LDA", lda_doc_scores),
            representative_documents(df, "LSA", lsa_doc_scores),
        ],
        ignore_index=True,
    )
    reps.to_csv(TABLE_DIR / "representative_complaints.csv", index=False)

    prevalence = pd.concat(
        [
            topic_prevalence("LDA", lda_doc_scores),
            topic_prevalence("LSA", lsa_doc_scores),
        ],
        ignore_index=True,
    )
    prevalence.to_csv(TABLE_DIR / "topic_prevalence.csv", index=False)

    selected_summary = {
        "selection_rule": (
            "highest NPMI coherence among tested topic counts; "
            "topic diversity used as first tie-breaker"
        ),
        "candidate_topic_counts": TOPIC_COUNTS,
        "selected_lda_topics": selected_lda_k,
        "selected_lsa_components": selected_lsa_k,
        "random_state": RANDOM_STATE,
        "vectorizer_parameters": {
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "tfidf_sublinear_tf": True,
        },
        "interpretation_notes": {
            "lda": (
                "LDA document-topic values are probabilities and can support "
                "direct topic-prevalence interpretation."
            ),
            "lsa": (
                "LSA components are signed latent dimensions, not probability "
                "distributions. Prevalence is reported only as document "
                "assignment using largest absolute component loading."
            ),
        },
    }
    (TABLE_DIR / "selected_model_summary.json").write_text(
        json.dumps(selected_summary, indent=2),
        encoding="utf-8",
    )

    save_diagnostic_figure(diagnostics)
    save_prevalence_figure(prevalence)

    print("\n=== SELECTED MODELS ===")
    print(f"LDA topics: {selected_lda_k}")
    print(f"LSA components: {selected_lsa_k}")
    print("\n=== OUTPUTS ===")
    print(TABLE_DIR / "vectorization_comparison.csv")
    print(TABLE_DIR / "model_diagnostics.csv")
    print(TABLE_DIR / "topic_terms.csv")
    print(TABLE_DIR / "representative_complaints.csv")
    print(TABLE_DIR / "topic_prevalence.csv")
    print(TABLE_DIR / "selected_model_summary.json")
    print(FIGURE_DIR / "topic_count_npmi_coherence.png")
    print(FIGURE_DIR / "lda_topic_prevalence.png")
    print(FIGURE_DIR / "lsa_topic_prevalence.png")


if __name__ == "__main__":
    main()
