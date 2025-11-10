"""
src/evaluator.py

Evaluates the performance of the Vibe Matcher system by measuring similarity scores and latency.
Provides functions for running multiple test queries and visualizing performance metrics.
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Callable, Tuple


def evaluate_queries(
    queries: List[str],
    embed_fn: Callable[[str], List[float]],
    match_fn: Callable[[List[float], pd.DataFrame, int], pd.DataFrame],
    df: pd.DataFrame,
    top_n: int = 3,
) -> Tuple[List[float], List[float]]:
    """Evaluate multiple queries for average similarity and latency.

    Parameters:
        queries: list of text queries
        embed_fn: function that generates embeddings from text
        match_fn: function that returns top-N matches
        df: product dataframe
        top_n: number of results per query

    Returns:
        Tuple of (average_scores, latencies)
    """
    avg_scores = []
    latencies = []

    for q in queries:
        print(f"\nQuery: {q}")
        start_time = time.time()

        query_emb = embed_fn(q)
        results = match_fn(query_emb, df, top_n=top_n)

        latency = time.time() - start_time
        avg_score = results["score"].mean()

        avg_scores.append(avg_score)
        latencies.append(latency)

        print(f"Average score: {avg_score:.3f} | Latency: {latency:.2f}s")

    return avg_scores, latencies


def plot_latency(queries: List[str], latencies: List[float]):
    """Plot latency per query using matplotlib."""
    plt.figure(figsize=(7, 4))
    plt.bar(queries, latencies)
    plt.title("Latency per Query")
    plt.ylabel("Seconds")
    plt.xlabel("Query")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_similarity(queries: List[str], avg_scores: List[float]):
    """Plot average similarity score per query."""
    plt.figure(figsize=(7, 4))
    plt.bar(queries, avg_scores)
    plt.title("Average Cosine Similarity per Query")
    plt.ylabel("Similarity Score")
    plt.xlabel("Query")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple demo with mock results
    sample_queries = ["urban chic", "cozy winter", "elegant evening"]
    latencies = [0.42, 0.39, 0.44]
    scores = [0.78, 0.81, 0.74]

    plot_latency(sample_queries, latencies)
    plot_similarity(sample_queries, scores)