# RegLIP: Regression-Based Contrastive Learning for Vision-Language Models

> Master's Thesis Research — [Your Name], [University], [Year]

## Abstract
[2-3 sentences: problem, approach, key result]

## Motivation
- Binary contrastive learning treats all negatives equally
- Semantically similar captions are penalized as hard negatives
- We propose using continuous text-text similarity as soft targets

## Method
### Architecture
[Keep your existing diagram — it's good]

### Loss Function
- **SigLIP (baseline):** Binary cross-entropy with log-sigmoid
- **RegLIP (proposed):** MSE on cosine similarities with soft targets from Qwen-8B

### Similarity Target Generation
[Keep your existing explanation]

## Experimental Setup
### Training
| Parameter | SigLIP | RegLIP |
|-----------|--------|--------|
| Base model | SigLIP-base-patch16-224 | SigLIP-base-patch16-224 |
| Training data | Flickr30K (31K pairs) | Flickr30K (31K pairs) |
| Batch size | 64 | 64 |
| Learning rate | 1e-4 | 1e-4 |
| Epochs | 10 | 10 |
| Frozen encoder | — | Qwen3-Embedding-8B |

### Evaluation Benchmarks
| Benchmark | Task | Metric |
|-----------|------|--------|
| Flickr30K | Image-Text Retrieval | R@1, R@5, R@10 |
| COCO | Image-Text Retrieval | R@1, R@5, R@10 |
| CIFAR-10 | Zero-shot Classification | Top-1, Top-5 Accuracy |
| CIFAR-100 | Zero-shot Classification | Top-1, Top-5 Accuracy |
| ImageNet-1K | Zero-shot Classification | Top-1, Top-5 Accuracy |

## Experiment Design

### Why Three Experiments?

Fine-tuning a pre-trained SigLIP model alone cannot conclusively prove a loss function works. The pre-trained weights were trained on billions of images using binary contrastive loss, creating an inherent bias toward SigLIP. We therefore design three complementary experiments, each answering a different research question.

### Experiment 1: Fine-Tuning Pre-Trained Models

```
Pre-trained SigLIP weights (google/siglip-base-patch16-224)
    ├──► Fine-tune with SigLIP loss on Flickr30K ──► Evaluate
    └──► Fine-tune with RegLIP loss on Flickr30K ──► Evaluate
```

**Research question:** Is RegLIP loss useful for adapting existing pre-trained models?

**Limitation:** The pre-trained model already learned representations optimized for binary matching. SigLIP fine-tuning continues with the same loss (natural advantage), while RegLIP must re-adapt the representations — using only 31K samples. If SigLIP outperforms RegLIP here, it may be due to pre-training bias, not loss quality.

---

### Experiment 2: Frozen Backbone (Projection Heads Only)

```
Pre-trained SigLIP (FROZEN vision + text encoders)
    ├──► Train projection heads with SigLIP loss ──► Evaluate
    └──► Train projection heads with RegLIP loss ──► Evaluate
```

**Research question:** Given identical features, does RegLIP loss learn better similarity mappings?

**Why this matters:** By freezing the encoders, both models start from the exact same feature representations. The only difference is how the projection heads are trained. This isolates the loss function effect from representation learning, removing the pre-training bias.

**Practical benefit:** Fast to train (only projection parameters update), low GPU memory.

---

### Experiment 3: Training from Scratch

```
Random initialization (same architecture)
    ├──► Train with SigLIP loss on CC3M (3M pairs) ──► Evaluate
    └──► Train with RegLIP loss on CC3M (3M pairs) ──► Evaluate
```

**Research question:** Does RegLIP loss learn fundamentally better vision-language representations?

**Why this matters:** No pre-trained weights, no bias toward either loss. Both models learn everything from scratch. This is the strongest evidence for or against the proposed loss function.

**Requirement:** A medium-scale dataset like CC3M (~3M image-text pairs) and significant compute time.

---

### How the Three Experiments Work Together

| Experiment | What It Isolates | If RegLIP Wins | If SigLIP Wins |
|------------|-----------------|----------------|----------------|
| Fine-tuning | Transfer learning ability | RegLIP is practical for real-world use | Pre-training bias may favor SigLIP (inconclusive) |
| Frozen backbone | Loss function quality on fixed features | Loss produces better similarity structure | Binary targets are sufficient for projection |
| From scratch | Fundamental representation learning | Strong evidence the loss is superior | Hypothesis is wrong (still publishable) |

**Possible outcomes and interpretations:**

- RegLIP wins **all three**: The loss function is conclusively better for vision-language learning.
- RegLIP wins **from scratch + frozen** but loses **fine-tuning**: Loss is better but pre-training bias hurts adaptation. Interesting finding about transfer learning.
- RegLIP wins **from scratch** only: Loss needs more data to shine. Suggests it's better for large-scale training.
- RegLIP wins **frozen** only: Loss learns better projections but doesn't improve full representation learning. Useful for lightweight adaptation.
- RegLIP loses **all three**: The hypothesis is wrong — binary targets are sufficient. Rigorous negative result, still publishable.

**Any outcome is publishable because the experimental design is thorough.**

---

### Thesis Experiments Chapter Structure

```
Chapter 4: Experiments

4.1 Experimental Setup
    4.1.1 Architecture and Base Model
    4.1.2 Training Hyperparameters
    4.1.3 Datasets (Flickr30K, CC3M, evaluation benchmarks)
    4.1.4 Evaluation Protocol and Metrics
    4.1.5 Reproducibility (seeds, hardware, software versions)

4.2 Experiment 1: Fine-Tuning Pre-Trained Models
    Table 4.1: Flickr30K Retrieval Results (Fine-Tuning)
    Table 4.2: Zero-Shot Classification Results (Fine-Tuning)
    Discussion: limitations of fine-tuning evaluation

4.3 Experiment 2: Frozen Backbone (Projection Head Training)
    Table 4.3: Flickr30K Retrieval Results (Frozen Backbone)
    Table 4.4: Zero-Shot Classification Results (Frozen Backbone)
    Discussion: isolating the loss function effect

4.4 Experiment 3: Training from Scratch on CC3M
    Table 4.5: Flickr30K Retrieval Results (From Scratch)
    Table 4.6: Zero-Shot Classification Results (From Scratch)
    Discussion: true loss function comparison

4.5 Cross-Experiment Analysis
    Table 4.7: Summary of all results across 3 settings
    Figure 4.1: Training curves comparison (loss convergence)
    Figure 4.2: Embedding space visualization (t-SNE / UMAP)
    Figure 4.3: Similarity target distribution analysis

4.6 Ablation Studies
    4.6.1 Frozen Encoder Choice (Qwen-8B vs smaller models)
    4.6.2 Loss Variants (MSE vs KL Divergence vs Smooth L1)
    4.6.3 Similarity Scaling Methods

4.7 Discussion
    - When does RegLIP help? When does it hurt?
    - Relationship between dataset size and loss benefit
    - Practical recommendations for practitioners
```

---

### Effort Estimate

| Experiment | Code Changes | Compute Required | Estimated Time |
|------------|-------------|-----------------|----------------|
| Fine-tuning | Already implemented | Low (Flickr30K, 31K pairs) | ~2-3 hours on A100 |
| Frozen backbone | Small (add freeze flag to config) | Low (only projection trains) | ~1-2 hours on A100 |
| From scratch | Medium (CC3M dataloader needed) | High (3M pairs, full training) | ~2-5 days on A100 |

---

## Results
[Tables comparing RegLIP vs SigLIP across all benchmarks]

## Reproduction

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (tested on NVIDIA A100 / V100)
- Access to a Qwen embedding API endpoint (for RegLIP training only)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd RegLIP

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Flickr30K (Training + Retrieval Evaluation)

Place the Flickr30K dataset under `data/flickr30k/`:

```
data/flickr30k/
├── Images/
│   ├── 1000092795.jpg
│   ├── 1000268201.jpg
│   └── ...
└── captions.txt
```

`captions.txt` format — each line: `filename,caption`

#### CIFAR-10 / CIFAR-100 (Zero-Shot Evaluation)

Downloaded automatically by `torchvision` on first run. No manual setup needed.

#### ImageNet-1K (Zero-Shot Evaluation)

Download from [Kaggle](https://www.kaggle.com/datasets/sautkin/imagenet1kvalid/code) and place under `data/imagenet-1k/`.

#### ImageNet-V2 (Zero-Shot Evaluation)

Download from [HuggingFace](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main) and place under `data/imagenetv2-matched-frequency-format-val/`.

### 3. Verify Dataset

```bash
python validate_dataset.py
# Or quick check only:
python validate_dataset.py --quick-only
```

### 4. Training

#### Train SigLIP (Baseline)

```bash
python training/train_siglip.py --config configs/siglip_config.yaml
```

#### Train RegLIP (Proposed)

Before training RegLIP, ensure the Qwen embedding API is reachable. Update `qwen_api_url` in `configs/reglip_config.yaml` if needed.

```bash
python training/train_reglip.py --config configs/reglip_config.yaml
```

#### Train Both (Side-by-Side Comparison)

```bash
python train_comparison.py
```

#### Monitor Training

```bash
tensorboard --logdir ./checkpoints/reglip/tensorboard   # RegLIP
tensorboard --logdir ./checkpoints/siglip/tensorboard   # SigLIP
tensorboard --logdir ./checkpoints                       # Both at once
# Open http://localhost:6006
```

### 5. Training Configuration Reference

Both models share the same base architecture. Key parameters:

| Parameter | SigLIP (`siglip_config.yaml`) | RegLIP (`reglip_config.yaml`) |
|-----------|-------------------------------|-------------------------------|
| Base model | `google/siglip-base-patch16-224` | `google/siglip-base-patch16-224` |
| Loss | `siglip_binary` | `reglip_regression` |
| Batch size | 64 | 64 |
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Scheduler | Cosine | Cosine |
| Warmup steps | 1000 | 1000 |
| Epochs | 10 | 10 |
| Image size | 224 | 224 |
| Max text length | 64 | 64 |
| Mixed precision | Yes | Yes |
| Frozen encoder | — | Qwen3-Embedding-8B |

Checkpoints are saved to:
- SigLIP: `./checkpoints/siglip/`
- RegLIP: `./checkpoints/reglip/`

### 6. Evaluation

#### Evaluate a Single Model

```bash
# RegLIP (fine-tuned)
bash evaluate.sh reglip

# SigLIP (fine-tuned)
bash evaluate.sh siglip

# Pre-trained baselines (no fine-tuning)
bash evaluate.sh reglip_base
bash evaluate.sh siglip_base
```

#### Compare RegLIP vs SigLIP

```bash
bash evaluate.sh compare
```

#### Task-Specific Evaluation

```bash
# Retrieval only (Flickr30K)
bash evaluate.sh retrieval

# Zero-shot classification only (CIFAR-10, CIFAR-100)
bash evaluate.sh zero_shot
```

#### ImageNet Benchmarks

```bash
# ImageNet-1K only
bash evaluate.sh imagenet

# All ImageNet variants (Val, V2, ReaL, ObjectNet)
bash evaluate.sh imagenet_all
```

#### MS-COCO Retrieval

```bash
bash evaluate.sh coco
```

Results are saved to `./results/` in JSON, CSV, and LaTeX formats.

### 7. Expected Output Structure

After training and evaluation, each model's outputs are self-contained:

```
RegLIP/
├── checkpoints/
│   ├── reglip/
│   │   ├── logs/              # Training text logs
│   │   ├── tensorboard/       # TensorBoard event files
│   │   ├── epoch_1.pth
│   │   ├── epoch_2.pth
│   │   ├── ...
│   │   ├── epoch_10.pth
│   │   └── best_model.pth
│   └── siglip/
│       ├── logs/
│       ├── tensorboard/
│       ├── epoch_1.pth
│       ├── ...
│       ├── epoch_10.pth
│       └── best_model.pth
└── results/
    ├── reglip_eval/           # RegLIP evaluation results
    ├── siglip_eval/           # SigLIP evaluation results
    └── comparison/            # Side-by-side comparison
```

### 8. Full Reproduction Script

To reproduce the complete experiment pipeline from scratch:

```bash
#!/bin/bash
set -e

echo "=== Step 1: Validate dataset ==="
python validate_dataset.py --quick-only

echo "=== Step 2: Train SigLIP baseline ==="
python training/train_siglip.py --config configs/siglip_config.yaml

echo "=== Step 3: Train RegLIP ==="
python training/train_reglip.py --config configs/reglip_config.yaml

echo "=== Step 4: Evaluate both models ==="
bash evaluate.sh reglip
bash evaluate.sh siglip

echo "=== Step 5: Generate comparison ==="
bash evaluate.sh compare

echo "=== Step 6: Zero-shot benchmarks ==="
bash evaluate.sh zero_shot

echo "=== Step 7: ImageNet evaluation (if available) ==="
if [ -d "data/imagenet-1k" ]; then
    bash evaluate.sh imagenet
else
    echo "Skipping ImageNet (data not found)"
fi

echo "=== Done! Results saved in ./results/ ==="
```

### 9. Hardware Requirements

| Setup | GPU Memory | Approximate Training Time (10 epochs) |
|-------|-----------|---------------------------------------|
| SigLIP (batch 64) | ~8 GB | ~2-3 hours on A100 |
| RegLIP (batch 64) | ~8 GB | ~3-5 hours on A100 |

Reduce `batch_size` in the config files if running on GPUs with less memory.