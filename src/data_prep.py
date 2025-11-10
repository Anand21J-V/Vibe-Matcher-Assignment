"""
src/data_prep.py

Creates and loads mock fashion dataset for the Vibe Matcher project.
Provides helper functions to save/load CSV and prepopulate example data.
Enhanced: auto-augments product descriptions with vibe context for better embedding quality.
"""

from pathlib import Path
import pandas as pd

# Project data directory (created automatically when saving)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = DATA_DIR / "fashion_data.csv"


def create_mock_data() -> pd.DataFrame:
    """Return a pandas DataFrame with around 10 mock fashion items.
    Each row contains:
      - name: str
      - desc: str (product description)
      - vibe: list[str]
    The `vibe` column is stored as a Python list (object dtype).
    """

    data = [
        {"name": "Boho Dress", "desc": "Flowy, earthy tones perfect for outdoor festivals and carefree days.", "vibe": ["boho", "free-spirited"]},
        {"name": "Streetwear Hoodie", "desc": "Bold colors and oversized fit for an energetic city vibe and skate culture.", "vibe": ["urban", "energetic", "chic"]},
        {"name": "Minimalist White Shirt", "desc": "Clean, crisp lines and lightweight fabric for a modern, simple look.", "vibe": ["minimalist", "chic", "elegant"]},
        {"name": "Cozy Knit Sweater", "desc": "Soft wool texture to keep you warm—perfect for relaxed winter evenings.", "vibe": ["cozy", "casual", "warm"]},
        {"name": "Elegant Black Blazer", "desc": "Sharp tailoring and subtle sheen for formal events or evening outings.", "vibe": ["formal", "elegant", "evening"]},
        {"name": "Sporty Tracksuit", "desc": "Lightweight, breathable fabric with athletic cut for active lifestyles.", "vibe": ["sporty", "casual", "energetic"]},
        {"name": "Retro Floral Top", "desc": "Colorful floral prints with a vintage silhouette for playful daytime looks.", "vibe": ["retro", "playful", "vintage"]},
        {"name": "Denim Jacket", "desc": "Classic blue denim with a rugged feel for a timeless casual aesthetic.", "vibe": ["casual", "vintage", "urban"]},
        {"name": "Summer Linen Pants", "desc": "Breathable, lightweight linen for cool comfort and natural elegance.", "vibe": ["summer", "minimalist", "chill"]},
        {"name": "Glam Sequin Dress", "desc": "Sparkly sequins and sleek fit for parties and bold evening fashion.", "vibe": ["glam", "evening", "party"]},
    ]

    df = pd.DataFrame(data)

    # Enrich each description with vibe-related context (semantic glue)
    df["desc"] = df.apply(
        lambda row: f"{row['desc']} This {row['name'].lower()} matches {', '.join(row['vibe'])} aesthetics and vibe.",
        axis=1
    )

    return df


def save_dataframe_to_csv(df: pd.DataFrame, path: Path = DEFAULT_CSV) -> Path:
    """Save DataFrame to CSV. `vibe` column will be stored as a JSON-like string.
    Returns the path saved.
    """
    df_copy = df.copy()
    df_copy["vibe"] = df_copy["vibe"].apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_copy.to_csv(path, index=False)
    return path


def load_dataframe_from_csv(path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the dataset from CSV and convert the `vibe` column back to list.
    If the file doesn't exist, this raises FileNotFoundError.
    """
    df = pd.read_csv(path)
    if "vibe" in df.columns:
        df["vibe"] = df["vibe"].fillna("").apply(lambda s: s.split(";") if isinstance(s, str) and s != "" else [])
    return df


if __name__ == "__main__":
    # When run directly, create default data and save to data/fashion_data.csv
    df = create_mock_data()
    saved = save_dataframe_to_csv(df)
    print(f"Saved enriched mock dataset to: {saved}")
