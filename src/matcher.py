"""
src/matcher.py

Handles cosine similarity and top-N product matching for the Vibe Matcher system.
Uses scikit-learn's cosine_similarity with normalization and optional vibe-based boosting
to produce more accurate, human-aligned results.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Optional


def compute_similarity(query_emb: List[float], item_embs: List[List[float]]) -> np.ndarray:
    """Compute cosine similarity between query embedding and all item embeddings."""
    query_emb = np.array(query_emb).reshape(1, -1)
    item_embs = np.stack(item_embs)

    # Normalize both query and item embeddings for true cosine distance
    query_emb = normalize(query_emb)
    item_embs = normalize(item_embs)

    sims = cosine_similarity(query_emb, item_embs)[0]
    return sims


def vibe_matcher(
    query_emb: List[float],
    df: pd.DataFrame,
    top_n: int = 3,
    query_text: Optional[str] = None,
    vibe_boost: float = 0.12
) -> pd.DataFrame:
    """
    Return top N most similar products given a query embedding.
    Optionally apply vibe keyword boosting for more intuitive matches.

    Parameters:
        query_emb: list or np.ndarray — query vector
        df: DataFrame — must contain an 'embedding' column
        top_n: int — number of top results to return
        query_text: optional query string for tag-based boosting
        vibe_boost: amount to boost similarity if vibe keyword matches query

    Returns:
        DataFrame sorted by final similarity score (descending)
    """
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column.")

    item_embs = df["embedding"].tolist()
    sims = compute_similarity(query_emb, item_embs)

    df_copy = df.copy()
    df_copy["score"] = sims

    # Apply optional vibe keyword boost
    if query_text:
        for idx, row in df_copy.iterrows():
            for vibe_tag in row["vibe"]:
                if vibe_tag.lower() in query_text.lower():
                    df_copy.at[idx, "score"] += vibe_boost
                    break  # avoid multiple boosts for same product

    top_matches = df_copy.sort_values(by="score", ascending=False).head(top_n)
    return top_matches[["name", "desc", "vibe", "score"]]


def pretty_print_results(results: pd.DataFrame):
    """Nicely print the matched products with scores."""
    print("\nTop Matching Products:\n" + "=" * 60)
    for _, row in results.iterrows():
        print(f"Product: {row['name']}")
        print(f"Vibe: {', '.join(row['vibe'])}")
        print(f"Score: {row['score']:.3f}")
        print(f"Desc: {row['desc']}")
        print("-" * 60)


if __name__ == "__main__":
    # Example demo using random embeddings
    dummy_df = pd.DataFrame({
        "name": ["Boho Dress", "Streetwear Hoodie", "Cozy Sweater"],
        "desc": ["Flowy dress", "Urban hoodie", "Soft warm sweater"],
        "vibe": [["boho"], ["urban"], ["cozy"]],
        "embedding": [np.random.rand(768).tolist() for _ in range(3)],  # 768 dims (mpnet)
    })

    query_emb = np.random.rand(768).tolist()
    query_text = "energetic urban chic"
    results = vibe_matcher(query_emb, dummy_df, query_text=query_text)
    pretty_print_results(results)
