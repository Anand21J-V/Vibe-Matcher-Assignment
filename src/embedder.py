"""
src/embedder.py

Handles text embedding generation for product descriptions and user queries.
Uses the free SentenceTransformer model "all-mpnet-base-v2" (offline, high-quality)
and supports caching + L2 normalization for improved similarity scoring.
"""

import os
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Initialize local embedding model
# (This will download the model the first time you run it)
print("Loading local embedding model: all-mpnet-base-v2 ...")
model = SentenceTransformer("all-mpnet-base-v2")
print("Model loaded successfully.")

# Paths for caching
CACHE_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "embeddings_cache.json"


# --- Helper: Load/Save cache ---
def _load_cache() -> dict:
    """Load embedding cache from JSON file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    """Save embedding cache to JSON file."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# --- Main embedding functions ---
def get_embedding(text: str) -> List[float]:
    """
    Get normalized embedding for a single text string using SentenceTransformer.
    Cached locally to reduce compute time.
    """
    cache = _load_cache()
    if text in cache:
        return cache[text]

    # Generate embedding
    vector = model.encode(text, normalize_embeddings=True).tolist()

    # Save to cache
    cache[text] = vector
    _save_cache(cache)
    return vector


def embed_dataframe(
    df: pd.DataFrame, text_column: str = "desc", embed_column: str = "embedding"
) -> pd.DataFrame:
    """
    Generate normalized embeddings for each row of a DataFrame.
    Adds a new column `embed_column` with the embedding vector for each text entry.
    Automatically caches embeddings for repeated runs.
    """
    df = df.copy()
    df[embed_column] = df[text_column].apply(get_embedding)
    return df


if __name__ == "__main__":
    # Example standalone usage
    sample_text = "Energetic urban chic style with vibrant tones."
    emb = get_embedding(sample_text)
    print(f"Embedding length: {len(emb)} | First 5 dims: {emb[:5]}")
