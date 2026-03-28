# Batch Text Embedding with Qwen API

This guide shows how to use the batch text embedding functionality to get embeddings from the Qwen API, compute dot products, and normalize results between 0 and 1.

## Quick Start

```python
from reglip.embedding_utils import QwenEmbeddingClient, batch_embedding_pipeline

# Initialize client
client = QwenEmbeddingClient(
    base_url="http://212.41.29.82:6010",
    model="Qwen/Qwen3-Embedding-8B"
)

# Your texts and possible labels
texts = ["hello my name is mahmoud", "I love machine learning"]
labels = ["greeting", "technology", "personal_information"]

# Get embeddings and predictions
text_embeddings, label_embeddings, predictions = batch_embedding_pipeline(
    texts=texts,
    labels=labels,
    client=client,
    batch_size=32,
    similarity_metric="dot_product",
    normalize=True,
    top_k=2
)

# Print results
for text, pred_list in zip(texts, predictions):
    print(f"Text: '{text}'")
    for label, score in pred_list:
        print(f"  {label}: {score:.4f}")
```

## Available Functions

### 1. QwenEmbeddingClient
Main client for interacting with the Qwen embedding API.

```python
client = QwenEmbeddingClient(
    base_url="http://212.41.29.82:6010",
    model="Qwen/Qwen3-Embedding-8B"
)

# Get embeddings for a batch of texts
embeddings = client.get_embeddings(
    texts=["text1", "text2", "text3"],
    batch_size=32,          # Process 32 texts per API call
    max_retries=3,          # Retry failed requests
    retry_delay=1.0         # Initial delay between retries
)
```

### 2. compute_similarity_matrix
Compute similarity between embeddings using dot product or cosine similarity.

```python
from reglip.embedding_utils import compute_similarity_matrix

# Dot product similarity (normalized to 0-1)
similarity_matrix = compute_similarity_matrix(
    embeddings1=text_embeddings,
    embeddings2=label_embeddings,  # Optional, uses embeddings1 if None
    normalize=True,                # Normalize to [0,1] range
    similarity_metric="dot_product"
)

# Cosine similarity (already normalized)
cosine_similarity = compute_similarity_matrix(
    embeddings1=text_embeddings,
    embeddings2=label_embeddings,
    normalize=False,               # Cosine is already normalized
    similarity_metric="cosine"
)
```

### 3. get_most_similar_labels
Find the most similar labels for each text.

```python
from reglip.embedding_utils import get_most_similar_labels

predictions = get_most_similar_labels(
    query_embeddings=text_embeddings,
    label_embeddings=label_embeddings,
    labels=["greeting", "technology", "weather"],
    top_k=2,                       # Return top 2 predictions
    similarity_metric="dot_product",
    normalize=True
)

# predictions is a list of lists: [[(label1, score1), (label2, score2)], ...]
```

### 4. batch_embedding_pipeline
Complete pipeline that handles everything.

```python
text_embeddings, label_embeddings, predictions = batch_embedding_pipeline(
    texts=["hello", "I love AI"],
    labels=["greeting", "technology"],
    client=client,                 # Optional, creates default if None
    batch_size=32,
    similarity_metric="dot_product",
    normalize=True,
    top_k=1
)
```

## Key Features

### ✅ Batch Processing
- Processes texts in configurable batches (default: 32)
- Progress bars for long operations
- Efficient memory usage

### ✅ Error Handling
- Automatic retries with exponential backoff
- Graceful failure handling
- Timeout protection (30s per request)

### ✅ Similarity Metrics
- **Dot Product**: Raw dot product with optional normalization to [0,1]
- **Cosine Similarity**: Normalized cosine similarity [-1,1] or [0,1]

### ✅ Normalization
- Dot product results normalized to [0,1] range: `(x - min) / (max - min)`
- Cosine similarity already normalized

### ✅ Flexible Input
- Works with numpy arrays or PyTorch tensors
- Automatic conversion between formats
- Supports both single and batch operations

## Example Output

```
Text embeddings shape: (5, 4096)
Label embeddings shape: (5, 4096)

Text 1: 'hello my name is mahmoud'
Top predictions:
  1. greeting: 1.0000
  2. personal_information: 0.7132

Text 2: 'I love machine learning and AI'
Top predictions:
  1. technology: 0.8236
  2. greeting: 0.2438
```

## Run the Example

```bash
python example_embedding.py
```

This will demonstrate:
1. Basic label prediction
2. Similarity matrix computation
3. Cosine vs dot product comparison
4. Single text embedding

## Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

The new requirements include:
- `requests>=2.28.0` for API calls
- `numpy>=1.21.0` for matrix operations
- `torch>=2.0.0` for tensor support
- `tqdm>=4.64.0` for progress bars 