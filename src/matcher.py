"""
src/matcher.py

Handles cosine similarity and top-N product matching for the Vibe Matcher system.
Uses scikit-learn's cosine_similarity to find the closest products to a query embedding.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def compute_similarity(query_emb: List[float], item_embs: List[List[float]]) -> np.ndarray:
    """Compute cosine similarity between query embedding and all item embeddings."""
    query_emb = np.array(query_emb).reshape(1, -1)
    item_embs = np.stack(item_embs)
    sims = cosine_similarity(query_emb, item_embs)[0]
    return sims


def vibe_matcher(query_emb: List[float], df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top N most similar products given a query embedding.

    Parameters:
        query_emb: list or np.ndarray — query vector
        df: DataFrame — must contain an 'embedding' column
        top_n: int — number of top results to return

    Returns:
        DataFrame sorted by similarity score (descending)
    """
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column.")

    item_embs = df["embedding"].tolist()
    sims = compute_similarity(query_emb, item_embs)

    df_copy = df.copy()
    df_copy["score"] = sims
    top_matches = df_copy.sort_values(by="score", ascending=False).head(top_n)
    return top_matches[["name", "desc", "vibe", "score"]]


def pretty_print_results(results: pd.DataFrame):
    """Nicely print the matched products with scores."""
    for _, row in results.iterrows():
        print(f"Product: {row['name']}\nVibe: {', '.join(row['vibe'])}\nScore: {row['score']:.3f}\nDesc: {row['desc']}\n{'-'*60}")


if __name__ == "__main__":
    # Example demo using random embeddings
    dummy_df = pd.DataFrame({
        "name": ["Boho Dress", "Streetwear Hoodie", "Cozy Sweater"],
        "desc": ["Flowy dress", "Urban hoodie", "Soft warm sweater"],
        "vibe": [["boho"], ["urban"], ["cozy"]],
        "embedding": [np.random.rand(384).tolist() for _ in range(3)],
    })

    query_emb = np.random.rand(384).tolist()
    results = vibe_matcher(query_emb, dummy_df)
    pretty_print_results(results)