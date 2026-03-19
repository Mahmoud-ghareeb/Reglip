# Chapter 4: Experiments

## 4.1 Experimental Setup

### 4.1.1 Architecture and Base Model

Both RegLIP and the SigLIP baseline share the same vision-language architecture from `google/siglip-base-patch16-224`:

| Component | Details |
|-----------|---------|
| Vision Encoder | ViT-Base/16, 12 layers, 12 heads, hidden size 768 |
| Text Encoder | 12 layers, 12 heads, hidden size 768, vocab 32K |
| Image Resolution | 224 x 224 |
| Patch Size | 16 x 16 |
| Projection Size | 768 |
| Max Text Length | 64 tokens |

The only architectural difference is that RegLIP additionally uses a **frozen Qwen3-Embedding-8B** text encoder to generate continuous similarity targets for training.

### 4.1.2 Training Hyperparameters

| Parameter | SigLIP (Baseline) | RegLIP (Proposed) |
|-----------|-------------------|-------------------|
| Base model | `google/siglip-base-patch16-224` | `google/siglip-base-patch16-224` |
| Loss function | Binary cross-entropy (SigLIP loss) | MSE regression loss |
| Training data | Flickr30K (~31K pairs) | Flickr30K (~31K pairs) |
| Batch size | 64 | 64 |
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | AdamW (weight decay 0.01) | AdamW (weight decay 0.01) |
| LR Scheduler | Cosine | Cosine |
| Warmup steps | 1000 | 1000 |
| Epochs | 10 | 10 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | Yes (FP16) | Yes (FP16) |
| Frozen encoder | -- | Qwen3-Embedding-8B |

### 4.1.3 Datasets

**Training:** Flickr30K (31,783 images, each with 5 captions)

**Evaluation Benchmarks:**

| Benchmark | Task | Metric | Source |
|-----------|------|--------|--------|
| CIFAR-10 | Zero-shot Classification | Top-1, Top-5 Accuracy | Torchvision (auto-download) |
| CIFAR-100 | Zero-shot Classification | Top-1, Top-5 Accuracy | Torchvision (auto-download) |
| ImageNet-1K | Zero-shot Classification | Top-1, Top-5 Accuracy | Manual download |
| ImageNet-V2 | Zero-shot Classification | Top-1, Top-5 Accuracy | HuggingFace |
| ImageNet-ReaL | Zero-shot Classification | Top-1, Top-5 Accuracy | Same split as ImageNet-1K with cleaned labels |

> **Note:** Flickr30K retrieval and MS-COCO retrieval benchmarks were not available for this evaluation round due to data path configuration. ObjectNet was similarly unavailable.

### 4.1.4 Evaluation Protocol

All models are evaluated using the same evaluation pipeline (`evaluate.sh`). Zero-shot classification uses text prompts constructed from class names and measures how well the model matches images to the correct class label without any task-specific training.

### 4.1.5 Reproducibility

- All experiments use the same random seeds
- Checkpoints are saved every epoch; the best model (lowest validation loss) is used for evaluation
- Hardware: NVIDIA GPU with CUDA, mixed-precision training enabled
- All configs are stored in `configs/reglip_config.yaml` and `configs/siglip_config.yaml`

---

## 4.2 Experiment Design Rationale

### Why Three Experiments?

Fine-tuning a pre-trained SigLIP model alone cannot conclusively prove that a loss function is superior. The pre-trained weights were trained on **billions** of images using binary contrastive loss, creating an inherent bias toward SigLIP. We therefore design three complementary experiments, each answering a different research question.

```
Experiment 1: Fine-tuning     ──► Can RegLIP adapt existing pre-trained models?
Experiment 2: Frozen backbone  ──► Does RegLIP learn better projections on fixed features?
Experiment 3: From scratch     ──► Does RegLIP learn fundamentally better representations?
```

| Experiment | What It Isolates | If RegLIP Wins | If SigLIP Wins |
|------------|-----------------|----------------|----------------|
| Fine-tuning | Transfer learning ability | RegLIP is practical for real-world use | Pre-training bias may favor SigLIP (inconclusive) |
| Frozen backbone | Loss function quality on fixed features | Loss produces better similarity structure | Binary targets are sufficient for projection |
| From scratch | Fundamental representation learning | Strong evidence the loss is superior | Hypothesis is wrong (still publishable as negative result) |

**Any outcome is publishable because the experimental design is thorough.**

---

## 4.3 Experiment 1: Fine-Tuning Pre-Trained Models

### Setup

```
Pre-trained SigLIP weights (google/siglip-base-patch16-224)
    ├──► Fine-tune with SigLIP loss on Flickr30K (10 epochs) ──► Evaluate
    └──► Fine-tune with RegLIP loss on Flickr30K (10 epochs) ──► Evaluate
```

**Research question:** Is RegLIP loss useful for adapting existing pre-trained models?

Both models start from the same pre-trained SigLIP checkpoint. The full model (vision encoder + text encoder) is fine-tuned end-to-end on Flickr30K. The only difference is the loss function used during fine-tuning.

### Results

#### Table 4.1: Zero-Shot Classification — CIFAR-10

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-------|:------------------:|:------------------:|
| Base (pre-trained, no fine-tuning) | 12.61 | 64.09 |
| SigLIP fine-tuned | 24.58 | 72.51 |
| **RegLIP fine-tuned** | **47.38** | **84.72** |

#### Table 4.2: Zero-Shot Classification — CIFAR-100

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-------|:------------------:|:------------------:|
| Base (pre-trained, no fine-tuning) | 1.28 | 7.52 |
| SigLIP fine-tuned | 5.13 | 11.72 |
| **RegLIP fine-tuned** | **8.54** | **22.69** |

#### Table 4.3: Zero-Shot Classification — ImageNet-1K

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-------|:------------------:|:------------------:|
| Base (pre-trained, no fine-tuning) | 0.42 | 1.92 |
| SigLIP fine-tuned | 1.40 | 3.22 |
| **RegLIP fine-tuned** | **2.19** | **6.72** |

#### Table 4.4: Zero-Shot Classification — ImageNet-V2

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-------|:------------------:|:------------------:|
| Base (pre-trained, no fine-tuning) | 0.39 | 1.54 |
| SigLIP fine-tuned | 1.28 | 3.12 |
| **RegLIP fine-tuned** | **2.15** | **6.78** |

#### Table 4.5: Zero-Shot Classification — ImageNet-ReaL

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|-------|:------------------:|:------------------:|
| Base (pre-trained, no fine-tuning) | 0.42 | 1.92 |
| SigLIP fine-tuned | 1.40 | 3.22 |
| **RegLIP fine-tuned** | **2.19** | **6.72** |

#### Table 4.6: Summary — All Zero-Shot Benchmarks (Top-1 Accuracy %)

| Dataset | Base Model | SigLIP Fine-tuned | RegLIP Fine-tuned | RegLIP Improvement over SigLIP |
|---------|:----------:|:------------------:|:------------------:|:------------------------------:|
| CIFAR-10 | 12.61 | 24.58 | **47.38** | +22.80 (+92.8%) |
| CIFAR-100 | 1.28 | 5.13 | **8.54** | +3.41 (+66.5%) |
| ImageNet-1K | 0.42 | 1.40 | **2.19** | +0.79 (+56.4%) |
| ImageNet-V2 | 0.39 | 1.28 | **2.15** | +0.87 (+68.0%) |
| ImageNet-ReaL | 0.42 | 1.40 | **2.19** | +0.79 (+56.4%) |

### Discussion

**Key findings from Experiment 1:**

1. **RegLIP outperforms SigLIP on all benchmarks.** Across every zero-shot classification dataset, the RegLIP fine-tuned model achieves higher accuracy than the SigLIP fine-tuned model, despite both starting from the same pre-trained weights.

2. **The improvement is substantial.** On CIFAR-10, RegLIP nearly doubles the top-1 accuracy compared to SigLIP (47.38% vs 24.58%), representing a +92.8% relative improvement. On CIFAR-100, the relative improvement is +66.5%.

3. **Both fine-tuned models improve over the base model.** Fine-tuning on Flickr30K (even with only ~31K image-text pairs) provides meaningful gains for zero-shot classification, confirming that both loss functions are learning useful vision-language representations.

4. **The absolute numbers are low** compared to state-of-the-art models trained on billions of pairs. This is expected: both models were fine-tuned on only ~31K pairs from Flickr30K, which is a very small dataset. The focus here is the *relative* comparison between loss functions, not absolute performance.

**Limitation:** The pre-trained SigLIP model was originally trained with binary contrastive loss on hundreds of millions of examples. Fine-tuning with SigLIP loss continues the same optimization trajectory, while RegLIP must re-adapt the learned representations to a regression-based objective. The fact that RegLIP still outperforms SigLIP despite this disadvantage is encouraging, but we cannot rule out that the pre-trained weights may interact differently with each loss. **Experiments 2 and 3 are needed to isolate the loss function's contribution.**

---

## 4.4 Experiment 2: Frozen Backbone (Projection Heads Only)

> **Status: Pending**

### Setup

```
Pre-trained SigLIP (FROZEN vision + text encoders)
    ├──► Train NEW projection heads with SigLIP loss ──► Evaluate
    └──► Train NEW projection heads with RegLIP loss ──► Evaluate
```

**Research question:** Given identical frozen features, does RegLIP loss learn better similarity mappings?

By freezing both encoders, both models operate on the exact same feature representations. The only difference is how the lightweight projection heads are trained. This isolates the loss function effect from representation learning.

**Code changes needed:** Add a `--freeze_encoders` flag to the training scripts so only the projection layers are trainable.

### Results

*To be filled after running the experiment.*

#### Table 4.7: Zero-Shot Classification — Frozen Backbone

| Dataset | SigLIP (Frozen) | RegLIP (Frozen) |
|---------|:---------------:|:---------------:|
| CIFAR-10 | -- | -- |
| CIFAR-100 | -- | -- |
| ImageNet-1K | -- | -- |
| ImageNet-V2 | -- | -- |
| ImageNet-ReaL | -- | -- |

### Discussion

*To be written after results are available.*

---

## 4.5 Experiment 3: Training from Scratch on CC3M

> **Status: Pending**

### Setup

```
Random initialization (same ViT-Base/16 architecture)
    ├──► Train from scratch with SigLIP loss on CC3M (~3M pairs) ──► Evaluate
    └──► Train from scratch with RegLIP loss on CC3M (~3M pairs) ──► Evaluate
```

**Research question:** Does RegLIP loss learn fundamentally better vision-language representations when there is no pre-training bias?

No pre-trained weights are used. Both models learn everything from scratch, providing the strongest evidence for or against the proposed loss function.

**Requirements:** CC3M dataset (~3M image-text pairs), significant compute time (~2-5 days on A100).

### Results

*To be filled after running the experiment.*

#### Table 4.8: Zero-Shot Classification — From Scratch

| Dataset | SigLIP (Scratch) | RegLIP (Scratch) |
|---------|:----------------:|:----------------:|
| CIFAR-10 | -- | -- |
| CIFAR-100 | -- | -- |
| ImageNet-1K | -- | -- |
| ImageNet-V2 | -- | -- |
| ImageNet-ReaL | -- | -- |

### Discussion

*To be written after results are available.*

---

## 4.6 Cross-Experiment Analysis

> **Status: Pending (requires all 3 experiments)**

#### Table 4.9: Summary Across All Settings (Top-1 Accuracy %)

| Dataset | Exp 1: Fine-tune (SigLIP) | Exp 1: Fine-tune (RegLIP) | Exp 2: Frozen (SigLIP) | Exp 2: Frozen (RegLIP) | Exp 3: Scratch (SigLIP) | Exp 3: Scratch (RegLIP) |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10 | 24.58 | **47.38** | -- | -- | -- | -- |
| CIFAR-100 | 5.13 | **8.54** | -- | -- | -- | -- |
| ImageNet-1K | 1.40 | **2.19** | -- | -- | -- | -- |
| ImageNet-V2 | 1.28 | **2.15** | -- | -- | -- | -- |
| ImageNet-ReaL | 1.40 | **2.19** | -- | -- | -- | -- |

### Possible Outcomes and Interpretations

- RegLIP wins **all three**: The loss function is conclusively better for vision-language learning.
- RegLIP wins **from scratch + frozen** but loses **fine-tuning**: Loss is better but pre-training bias hurts adaptation. Interesting finding about transfer learning.
- RegLIP wins **from scratch** only: Loss needs more data to shine. Suggests it's better for large-scale training.
- RegLIP wins **frozen** only: Loss learns better projections but doesn't improve full representation learning. Useful for lightweight adaptation.
- RegLIP loses **all three**: The hypothesis is wrong — binary targets are sufficient. Rigorous negative result, still publishable.

---

## 4.7 Ablation Studies

> **Status: Planned**

Potential ablation directions to explore after the main experiments:

- **4.7.1 Frozen Encoder Choice:** Compare Qwen-8B vs smaller models (e.g., Qwen-1.5B, all-MiniLM) as the similarity target generator.
- **4.7.2 Loss Variants:** MSE vs KL Divergence vs Smooth L1 for the regression objective.
- **4.7.3 Similarity Scaling Methods:** Different ways to map raw cosine similarity to target range.

---

## How to Reproduce

### Experiment 1 (Fine-Tuning)

```bash
# Train SigLIP baseline
python training/train_siglip.py --config configs/siglip_config.yaml

# Train RegLIP
python training/train_reglip.py --config configs/reglip_config.yaml

# Evaluate all three (base, siglip fine-tuned, reglip fine-tuned)
bash evaluate.sh reglip_base   # pre-trained baseline
bash evaluate.sh siglip        # siglip fine-tuned
bash evaluate.sh reglip        # reglip fine-tuned
```

### Experiment 2 (Frozen Backbone)

```bash
# To be implemented: add --freeze_encoders flag
```

### Experiment 3 (From Scratch)

```bash
# To be implemented: requires CC3M dataset and random initialization config
```
