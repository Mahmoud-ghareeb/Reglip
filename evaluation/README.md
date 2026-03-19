# Evaluation Framework for RegLIP and SigLIP

This directory contains a **unified, configurable evaluation framework** for:

- Image–Text **retrieval** (e.g. Flickr30K)
- **Zero-shot classification** (e.g. CIFAR-10 / CIFAR-100)
- **Model comparison** between RegLIP and SigLIP

The design is generic so you can plug in new datasets and tasks later.

---

## Directory Structure

```text
evaluation/
├── __init__.py              # Public API (Evaluator, metrics helpers)
├── evaluator.py             # Main Evaluator class + compare_models()
├── metrics.py               # R@K, top-K accuracy, similarity stats
├── utils.py                 # Feature extraction + result formatting/saving
├── tasks/
│   ├── __init__.py
│   ├── base.py              # BaseTask abstract class
│   ├── retrieval.py         # Image–Text retrieval task
│   └── zero_shot.py         # Zero-shot classification task
└── datasets/
    ├── __init__.py
    ├── base.py              # BaseEvalDataset abstract class
    ├── flickr30k.py         # Flickr30K retrieval dataset
    └── cifar.py             # CIFAR-10 / CIFAR-100 classification datasets
```

Related files outside this folder:

```text
scripts/run_evaluation.py    # CLI entrypoint
configs/eval_config.yaml     # Example evaluation configuration
evaluate.sh                  # Convenience shell script
results/                     # Default output directory
```

---

## Supported Tasks

### 1. Image–Text Retrieval (`retrieval` task)

Implemented in `tasks/retrieval.py` using:

- **Dataset**: `Flickr30KRetrievalDataset` (`datasets/flickr30k.py`)
- **Inputs**:
  - Images + a **single caption per image** (first caption in `captions.txt`)
  - Uses the **same split logic** as training:
    - `train`: first 80% of images
    - `val`: next 10%
    - `test`: last 10%
- **Process**:
  1. Extract **image features** for all images with `model.get_image_features`
  2. Extract **text features** for all captions with `model.get_text_features`
  3. Compute a similarity matrix \( S \in \mathbb{R}^{N_{\text{img}} \times N_{\text{text}}} \)
  4. Compute retrieval metrics from this matrix

**Metrics (in `metrics.py`):**

- Image → Text:
  - `i2t_r1`, `i2t_r5`, `i2t_r10`  (Recall@K, %)
- Text → Image:
  - `t2i_r1`, `t2i_r5`, `t2i_r10`  (Recall@K, %)
- Aggregate:
  - `mean_recall` (mean of all 6 recalls)
  - `i2t_median_rank`, `t2i_median_rank`
- Optional similarity statistics:
  - `sim_diagonal_mean`, `sim_off_diagonal_mean`, `sim_gap`, etc.

### 2. Zero-Shot Classification (`zero_shot` task)

Implemented in `tasks/zero_shot.py` for:

- `CIFAR10Dataset` and `CIFAR100Dataset` (`datasets/cifar.py`)

**Process:**

1. Get **class names** from the dataset (`get_class_names()`).
2. Turn class names into **prompts** using templates, e.g.:
   - `"a photo of a {}"`, `"a picture of a {}"`, `"an image of a {}"`, `"{}"`.
3. Encode all prompts once with `model.get_text_features` to get **class embeddings**.
4. For each image:
   - Extract image features
   - Compute similarity to all class embeddings
   - Predict the class with highest similarity (or top-K)

**Metrics (in `metrics.py`):**

- `top1_accuracy` (Top-1 accuracy, %)
- `top5_accuracy` (Top-5 accuracy, %)

**Prompt configuration (fully configurable):**

- CLI:
  - `--templates "a photo of a {}" "an image of {}"` to override defaults
  - `--no_ensemble` to disable template ensembling and use a single template
- In code:
  - `ZeroShotTask.DEFAULT_TEMPLATES` can be changed for your experiments

---

## Evaluator API

The central entrypoint is `Evaluator` in `evaluation/evaluator.py`.

### Creating an Evaluator

```python
from evaluation import Evaluator

evaluator = Evaluator(
    model=my_model,            # must implement get_image_features / get_text_features
    tokenizer=my_tokenizer,    # text tokenizer matching the model
    image_processor=my_img_proc,
    device="cuda",
)
```

### Single Task + Dataset

```python
metrics = evaluator.evaluate(
    task="retrieval",          # or "zero_shot"
    dataset_name="flickr30k",  # or "cifar10", "cifar100"
    dataset_kwargs={
        "data_root": "/home/mahmoud/RegLIP/data/flickr30k",
        "split": "test",
    },
    task_kwargs={
        "batch_size": 64,
        "k_values": [1, 5, 10],
    },
)
```

### Multiple Tasks / Datasets

```python
results = evaluator.evaluate_all(
    datasets=["flickr30k", "cifar10"],
    tasks=["retrieval", "zero_shot"],
    dataset_kwargs={
        "flickr30k": {"data_root": "/home/mahmoud/RegLIP/data/flickr30k", "split": "test"},
        "cifar10":   {"data_root": "./data", "split": "test"},
    },
    task_kwargs={
        "retrieval": {"batch_size": 64, "k_values": [1, 5, 10], "compute_stats": True},
        "zero_shot": {
            "batch_size": 64,
            "k_values": [1, 5],
            "templates": ["a photo of a {}", "{}"],
            "use_ensemble": True,
        },
    },
)
```

The nested result structure is:

```python
{
  "flickr30k": {
    "retrieval": { ...metrics... }
  },
  "cifar10": {
    "zero_shot": { ...metrics... }
  }
}
```

---

## CLI: `scripts/run_evaluation.py`

The CLI wraps `Evaluator` for easier experiments.

### Basic Examples

```bash
# 1) Evaluate a single model on all tasks/datasets
python scripts/run_evaluation.py \
  --checkpoint checkpoints/reglip/best_model.pth \
  --task all \
  --dataset all \
  --data_root /home/mahmoud/RegLIP/data/flickr30k \
  --output results/reglip_eval \
  --format json csv latex

# 2) Retrieval only on Flickr30K test
python scripts/run_evaluation.py \
  --checkpoint checkpoints/reglip/best_model.pth \
  --task retrieval \
  --dataset flickr30k \
  --data_root /home/mahmoud/RegLIP/data/flickr30k \
  --output results/reglip_retrieval

# 3) Zero-shot only on CIFAR-10 with custom templates
python scripts/run_evaluation.py \
  --checkpoint checkpoints/reglip/best_model.pth \
  --task zero_shot \
  --dataset cifar10 \
  --templates "a photo of a {}" "{}" \
  --output results/reglip_cifar10_zero_shot
```

Key arguments:

- `--checkpoint`: one or more model checkpoints
- `--model_type`: `reglip`, `siglip`, or `auto` (default)
- `--task`: `retrieval`, `zero_shot`, or `all`
- `--dataset`: `flickr30k`, `cifar10`, `cifar100`, or `all`
- `--data_root`: Flickr30K root (defaults to your project path)
- `--cifar_root`: CIFAR data root (downloads via torchvision)
- `--templates`: custom zero-shot prompt templates
- `--no_ensemble`: disable prompt ensembling
- `--output`: base path for results (without extension)
- `--format`: any of `json`, `csv`, `latex`

---

## Comparison Mode (RegLIP vs SigLIP)

`compare_models` in `evaluator.py` lets you evaluate and compare multiple models in **one command**.

### Example (via CLI)

```bash
python scripts/run_evaluation.py \
  --checkpoint checkpoints/reglip/best_model.pth checkpoints/siglip/best_model.pth \
  --model_names RegLIP SigLIP \
  --compare \
  --task all \
  --dataset all \
  --data_root /home/mahmoud/RegLIP/data/flickr30k \
  --output results/comparison \
  --format json csv latex
```

This will:

1. Run all selected tasks/datasets for **each model**.
2. Print a **side-by-side comparison table** to the console.
3. Save comparison results to:
   - `results/comparison_reglip.json`, `results/comparison_siglip.json`
   - `results/comparison_comparison.json` / `.csv` / `.tex` (combined view)

The comparison table shows, for example:

- Retrieval:
  - `I2T R@1`, `I2T R@5`, `I2T R@10`
  - `T2I R@1`, `Mean Recall`
- Zero-shot:
  - `Top-1 Acc`, `Top-5 Acc`

---

## Convenience Script: `evaluate.sh`

`evaluate.sh` wraps common CLI invocations:

```bash
# Evaluate RegLIP on all tasks/datasets
./evaluate.sh reglip

# Evaluate SigLIP
./evaluate.sh siglip

# Compare RegLIP vs SigLIP
./evaluate.sh compare

# Retrieval only (Flickr30K)
./evaluate.sh retrieval

# Zero-shot only (CIFAR-10/100)
./evaluate.sh zero_shot
```

Edit the defaults inside `evaluate.sh` if you move checkpoints or data:

- `REGLIP_CHECKPOINT`
- `SIGLIP_CHECKPOINT`
- `DATA_ROOT`
- `OUTPUT_DIR`

---

## Output Formats

All outputs are written under `results/` (configurable via CLI).

For a single model:

- `eval.json` – full nested metrics
- `eval.csv` – flat table (model, dataset, task, metric, value)
- `eval.tex` – LaTeX tables for retrieval + zero-shot (ready for thesis)

For comparisons:

- `comparison_reglip.*`, `comparison_siglip.*` – per-model results
- `comparison_comparison.*` – combined comparison across models

You can change which formats are written using `--format`:

```bash
--format json           # JSON only
--format json csv       # JSON + CSV
--format json csv latex # JSON + CSV + LaTeX
```

---

## Adding New Datasets or Tasks (Extensibility)

- **New retrieval dataset**:
  - Inherit from `BaseEvalDataset`
  - Implement `__len__` and `__getitem__` to return `{ 'pixel_values', 'caption', 'image_id' }`
  - Register in `Evaluator.DATASETS` with `task_type: "retrieval"`

- **New classification dataset**:
  - Inherit from `BaseEvalDataset`
  - Implement `__getitem__` to return `{ 'pixel_values', 'label' }`
  - Implement `get_class_names()`
  - Register in `Evaluator.DATASETS` with `task_type: "classification"`

- **New task**:
  - Inherit from `BaseTask`
  - Implement `run(dataset, **kwargs) -> Dict[str, float]`
  - Register in `Evaluator.TASKS`

This design lets you keep **one evaluation interface** while expanding your experiments.

