# Vibe Matcher - Fashion Recommendation System

A lightweight, offline fashion recommendation system that maps user-provided "vibe" queries (e.g., "energetic urban chic") to the top-3 matching fashion products from a curated catalog using semantic embeddings and cosine similarity.

## 🎯 Overview

**Vibe Matcher** transforms subjective fashion preferences into data-driven recommendations. Instead of rigid category-based search, users can describe their mood, aesthetic, or event, and the system returns products that match their vibe.

### Key Features

- **Semantic Search**: Uses sentence embeddings to understand natural language queries
- **Offline & Free**: No API keys or external services required
- **Fast & Lightweight**: Cached embeddings and efficient similarity computation
- **Interpretable Results**: Clear similarity scores and product rankings
- **Extensible**: Easy to upgrade to FAISS, Pinecone, or cloud embeddings

## 📋 Technical Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | `all-mpnet-base-v2` (SentenceTransformers) |
| **Similarity** | Cosine similarity (scikit-learn) |
| **Data Storage** | CSV + JSON cache |
| **Language** | Python 3.10+ |
| **Notebook** | Jupyter/Colab compatible |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vibe-matcher.git
cd vibe-matcher

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

```
vibe-matcher/
├── data/
│   ├── fashion_data.csv          # Product catalog
│   └── embeddings_cache.json     # Cached embeddings
├── src/
│   ├── data_prep.py              # Data loading/creation
│   ├── embedder.py               # Embedding generation
│   ├── matcher.py                # Similarity & ranking
│   └── evaluator.py              # Performance metrics
├── assets/
│   └── latency_plot.png          # Visualization outputs
├── vibe_matcher.ipynb            # Main demo notebook
├── requirements.txt
└── README.md
```

### 3. Running the System

#### Option A: Jupyter Notebook (Recommended)

```bash
jupyter notebook vibe_matcher.ipynb
```

Follow the notebook cells to:
1. Load/create the fashion dataset
2. Generate embeddings
3. Query the system with custom vibes
4. Evaluate performance

#### Option B: Python Scripts

```python
from src.data_prep import load_dataframe_from_csv
from src.embedder import embed_dataframe, get_embedding
from src.matcher import vibe_matcher

# Load data
df = load_dataframe_from_csv()

# Generate embeddings (cached after first run)
df = embed_dataframe(df)

# Query
query_emb = get_embedding("energetic urban chic")
results = vibe_matcher(query_emb, df, top_n=3)
print(results)
```

## 🔍 Usage Examples

### Basic Query

```python
query = "cozy winter aesthetic"
results = match_query_text(query, top_n=3)
pretty_print_results(results)
```

**Output:**
```
Top Matching Products:
============================================================
Product: Cozy Knit Sweater
Vibe: cozy, casual, warm
Score: 0.685
Desc: Soft wool texture to keep you warm—perfect for relaxed winter evenings...
```

### Batch Evaluation

```python
test_queries = [
    "energetic urban chic",
    "warm cozy winter look",
    "sparkly evening glam"
]

avg_scores, latencies = evaluate_queries(
    queries=test_queries,
    embed_fn=get_embedding,
    match_fn=vibe_matcher,
    df=df,
    top_n=3
)

plot_latency(test_queries, latencies)
plot_similarity(test_queries, avg_scores)
```

## 📊 Dataset

The default dataset includes **10 fashion items** with:
- **Name**: Product identifier
- **Description**: Enhanced with vibe keywords for better matching
- **Vibe Tags**: `["urban", "cozy", "elegant", etc.]`
- **Embedding**: 768-dim vector from `all-mpnet-base-v2`

### Sample Data

| Name | Vibe Tags | Description Snippet |
|------|-----------|---------------------|
| Boho Dress | boho, free-spirited | Flowy, earthy tones perfect for outdoor festivals... |
| Streetwear Hoodie | urban, energetic, chic | Bold colors and oversized fit for city vibes... |
| Cozy Knit Sweater | cozy, casual, warm | Soft wool texture for relaxed winter evenings... |

### Extending the Dataset

```python
# Add custom products
new_items = [
    {
        "name": "Vintage Denim Jacket",
        "desc": "Classic 90s denim with distressed details",
        "vibe": ["vintage", "casual", "retro"]
    }
]

df = pd.concat([df, pd.DataFrame(new_items)], ignore_index=True)
df = embed_dataframe(df)
save_dataframe_to_csv(df)
```

## ⚙️ Configuration

### Embedding Model

To switch models (requires model download):

```python
# In src/embedder.py
model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster, smaller
# or
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")  # Multilingual
```

### Similarity Boosting

Adjust vibe-based keyword boosting:

```python
results = vibe_matcher(
    query_emb, 
    df, 
    top_n=3,
    query_text="urban chic",
    vibe_boost=0.15  # Increase boost strength
)
```

### Fallback Threshold

```python
THRESHOLD = 0.30

def match_with_fallback(query, top_n=3):
    results = match_query_text(query, top_n)
    if results['score'].max() < THRESHOLD:
        print("⚠️ No strong matches found. Try broader terms.")
    return results
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Average Latency** | ~0.42s per query |
| **Embedding Dimension** | 768 |
| **Top-3 Avg Similarity** | 0.65-0.85 (good matches) |
| **Cache Hit Rate** | ~100% after warmup |

### Evaluation Outputs

```python
# Evaluate multiple queries
avg_scores, latencies = evaluate_queries(
    queries=["urban chic", "cozy winter", "elegant evening"],
    embed_fn=get_embedding,
    match_fn=vibe_matcher,
    df=df
)

# Visualizations saved to assets/
plot_latency(queries, latencies)
plot_similarity(queries, avg_scores)
```

## 🛠️ Advanced Features

### Custom Query Expansion

```python
def expand_query(vibe_query: str) -> str:
    """Add context to improve embedding quality"""
    return f"I am looking for a fashion outfit that feels {vibe_query}."

query_emb = get_embedding(expand_query("minimalist chic"))
```

### Hybrid Ranking (Semantic + Keyword)

```python
# Combines embedding similarity with vibe tag matching
results = vibe_matcher(
    query_emb,
    df,
    query_text="urban energetic",  # Boosts items with "urban"/"energetic" tags
    vibe_boost=0.12
)
```

## 📝 API Reference

### Core Functions

#### `embed_dataframe(df, text_column, embed_column)`
Generates and caches embeddings for all products.

**Parameters:**
- `df` (DataFrame): Product data
- `text_column` (str): Column to embed (default: "desc")
- `embed_column` (str): Output column name (default: "embedding")

**Returns:** DataFrame with embeddings

---

#### `vibe_matcher(query_emb, df, top_n, query_text, vibe_boost)`
Ranks products by similarity to query.

**Parameters:**
- `query_emb` (List[float]): Query embedding vector
- `df` (DataFrame): Product catalog with embeddings
- `top_n` (int): Number of results (default: 3)
- `query_text` (str, optional): For vibe boosting
- `vibe_boost` (float): Boost amount (default: 0.12)

**Returns:** DataFrame of top matches with scores

---

#### `evaluate_queries(queries, embed_fn, match_fn, df, top_n)`
Batch evaluation with metrics.

**Returns:** Tuple of (avg_scores, latencies)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add FAISS indexing for large catalogs (10k+ items)
- [ ] Multi-modal support (text + images)
- [ ] User preference learning
- [ ] A/B testing framework
- [ ] REST API wrapper

```bash
# Fork the repo, create a branch
git checkout -b feature/amazing-feature

# Make changes, commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SentenceTransformers**: High-quality semantic embeddings
- **scikit-learn**: Efficient similarity computations
- Fashion dataset inspired by e-commerce catalogs

## 📧 Contact

For questions or collaboration:
- **Email**: anandvishwakarma21j@gmail.comg
- **GitHub**: https://github.com/Anand21J-V
- **LinkedIn**: www.linkedin.com/in/anand-vishwakarma-07110a366

---

**Built with ❤️ for better fashion discovery through AI**
