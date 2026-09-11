"""Optional robustness check for the final Phase 3 submission.

Run after the core pipeline. The script evaluates whether the selected topic
structures are sensitive to the random seed without changing the submitted
primary models (which use random_state=42).

It compares:
- LDA with k=6 and k=8 across seeds 7, 21, 42, 84, 123
- LSA with k=4 across the same seeds

Outputs are written to outputs/tables/stability_check.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "complaints_preprocessed.csv"
OUTPUT_PATH = ROOT / "outputs" / "tables" / "stability_check.csv"
TEXT_COLUMN = "clean_text"
SEEDS = [7, 21, 42, 84, 123]
TOP_WORDS = 10


def get_top_indices(components: np.ndarray) -> list[list[int]]:
    return [np.argsort(row)[::-1][:TOP_WORDS].tolist() for row in components]


def topic_diversity(top_indices: list[list[int]]) -> float:
    flat = [idx for topic in top_indices for idx in topic]
    return len(set(flat)) / len(flat)


def binary_matrix(matrix):
    result = matrix.copy().tocsr()
    result.data = np.ones_like(result.data)
    return result


def npmi_coherence(binary_doc_term, top_indices: list[list[int]], eps: float = 1e-12) -> float:
    n_docs = binary_doc_term.shape[0]
    unique_terms = sorted({idx for topic in top_indices for idx in topic})
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
    return float(np.mean(topic_scores))


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run python run_analysis.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    texts = df[TEXT_COLUMN].fillna("").astype(str)
    texts = texts.loc[texts.str.strip().ne("")]

    bow_vectorizer = CountVectorizer(
        min_df=5, max_df=0.95, max_features=5000, ngram_range=(1, 2)
    )
    tfidf_vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_bow = bow_vectorizer.fit_transform(texts)
    x_tfidf = tfidf_vectorizer.fit_transform(texts)
    binary_bow = binary_matrix(x_bow)

    rows = []
    for seed in SEEDS:
        for k in (6, 8):
            model = LatentDirichletAllocation(
                n_components=k,
                learning_method="online",
                max_iter=5,
                batch_size=256,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(x_bow)
            top_indices = get_top_indices(model.components_)
            rows.append(
                {
                    "method": "LDA",
                    "seed": seed,
                    "n_topics": k,
                    "npmi_coherence": npmi_coherence(binary_bow, top_indices),
                    "topic_diversity": topic_diversity(top_indices),
                    "perplexity": float(model.perplexity(x_bow)),
                    "explained_variance": np.nan,
                }
            )

        lsa = TruncatedSVD(
            n_components=4,
            algorithm="randomized",
            n_iter=10,
            random_state=seed,
        )
        lsa.fit(x_tfidf)
        top_indices = get_top_indices(lsa.components_)
        rows.append(
            {
                "method": "LSA",
                "seed": seed,
                "n_topics": 4,
                "npmi_coherence": npmi_coherence(binary_bow, top_indices),
                "topic_diversity": topic_diversity(top_indices),
                "perplexity": np.nan,
                "explained_variance": float(np.sum(lsa.explained_variance_ratio_)),
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(result.to_string(index=False))
    print("\nMean LDA diagnostics by topic count:")
    print(
        result.loc[result["method"] == "LDA"]
        .groupby("n_topics")[["npmi_coherence", "topic_diversity", "perplexity"]]
        .mean()
        .to_string()
    )
    print(f"\n[OK] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
