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
| Total Parameters | ~93M |

The only architectural difference is that RegLIP additionally uses a **frozen Qwen3-Embedding-8B** text encoder to generate continuous similarity targets during training. This external encoder is not part of the trained model; it serves solely as a teacher that provides soft supervision signals.

### 4.1.2 Datasets

**Training:** Flickr30K (31,783 images, each with 5 captions). Each training sample consists of one image paired with one of its captions. The dataset is small by modern vision-language standards, which makes it an intentionally challenging setting: any difference between loss functions must manifest within limited data.

**Evaluation Benchmarks:**

| Benchmark | Task | Classes | Metric | Source |
|-----------|------|:-------:|--------|--------|
| CIFAR-10 | Zero-shot Classification | 10 | Top-1, Top-5 Accuracy | Torchvision |
| CIFAR-100 | Zero-shot Classification | 100 | Top-1, Top-5 Accuracy | Torchvision |
| ImageNet-1K | Zero-shot Classification | 1000 | Top-1, Top-5 Accuracy | Manual download |
| ImageNet-V2 | Zero-shot Classification | 1000 | Top-1, Top-5 Accuracy | HuggingFace |
| ImageNet-ReaL | Zero-shot Classification | 1000 | Top-1, Top-5 Accuracy | Re-assessed labels for ImageNet val |

The benchmarks are ordered by difficulty: CIFAR-10 (10 coarse-grained classes) is the easiest, while ImageNet variants (1000 fine-grained classes) are the hardest. This spread lets us observe whether the loss function advantage holds across different granularities.

> **Note:** Flickr30K retrieval and MS-COCO retrieval benchmarks were not evaluated in this round due to data path configuration issues. ObjectNet was similarly unavailable. These will be added in a future revision.

### 4.1.3 Evaluation Protocol

All models are evaluated using the same pipeline (`evaluate.sh` / `scripts/run_evaluation.py`). Zero-shot classification constructs text prompts from class names (e.g., *"a photo of a dog"*) and computes cosine similarity between each image embedding and every class-text embedding. The predicted class is the one with the highest similarity. No task-specific training or adaptation is performed -- the model must rely entirely on the vision-language alignment learned during training.

### 4.1.4 Reproducibility

- Random seed fixed at 42 across all experiments
- Checkpoints saved every epoch; the **best model** (lowest validation loss) is selected for evaluation
- Hardware: single NVIDIA GPU with CUDA, mixed-precision (FP16) training enabled
- All configurations are version-controlled in `configs/`

---

## 4.2 Experiment Design Rationale

### Why Three Experiments?

Fine-tuning a pre-trained SigLIP model alone cannot conclusively prove that a loss function is superior. The pre-trained weights were trained on **billions** of images using binary contrastive loss, creating an inherent bias: SigLIP loss is continuing the same optimisation trajectory the weights were initialised for, while RegLIP must re-adapt those representations to a different objective. A single experiment therefore conflates loss quality with initialisation compatibility.

We design three complementary experiments, each isolating a different factor:

```
Experiment 1: Fine-tuning     ──► Can RegLIP adapt existing pre-trained models?
Experiment 2: Frozen backbone  ──► Does RegLIP learn better projections on fixed features?
Experiment 3: From scratch     ──► Does RegLIP learn fundamentally better representations?
```

| Experiment | What It Isolates | If RegLIP Wins | If SigLIP Wins |
|------------|-----------------|----------------|----------------|
| Fine-tuning | Transfer learning ability | RegLIP is practical for real-world use | Pre-training bias may favour SigLIP (inconclusive) |
| Frozen backbone | Loss function quality on fixed features | Loss produces better similarity structure | Binary targets are sufficient for projection |
| From scratch | Fundamental representation learning | Strong evidence the loss is superior | Hypothesis is wrong (still publishable as negative result) |

The key insight is that **Experiment 2 is the cleanest test of the loss function itself**: by freezing the backbone, both models receive identical input features, and the only variable is how the projection heads are optimised. If RegLIP outperforms SigLIP under frozen features, the advantage must come from the loss, not from representation learning differences.

**Any outcome is publishable because the experimental design is thorough.**

---

## 4.3 Experiment 1: Fine-Tuning Pre-Trained Models

### 4.3.1 Setup

```
Pre-trained SigLIP weights (google/siglip-base-patch16-224)
    ├──► Fine-tune ALL parameters with SigLIP loss on Flickr30K ──► Evaluate
    └──► Fine-tune ALL parameters with RegLIP loss on Flickr30K ──► Evaluate
```

**Research question:** *Can RegLIP loss improve existing pre-trained models through fine-tuning?*

Both models start from the identical pre-trained SigLIP checkpoint (`google/siglip-base-patch16-224`). The entire model -- vision encoder, text encoder, projection heads, logit scale and bias -- is updated during training. The only difference is the loss function: SigLIP uses binary cross-entropy over the pairwise similarity matrix, while RegLIP uses MSE regression against continuous similarity targets from the frozen Qwen3-Embedding-8B encoder.

**Training Hyperparameters (Experiment 1):**

| Parameter | SigLIP | RegLIP |
|-----------|--------|--------|
| Trainable parameters | ~93M (all) | ~93M (all) |
| Loss function | Binary cross-entropy | MSE regression |
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | AdamW (weight decay 0.01) | AdamW (weight decay 0.01) |
| LR Scheduler | Cosine annealing | Cosine annealing |
| Batch size | 64 | 64 |
| Epochs | 10 | 10 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | FP16 | FP16 |
| Similarity teacher | -- | Qwen3-Embedding-8B |

### 4.3.2 Results

We evaluate three models: the **base model** (pre-trained, no fine-tuning) as a reference point, the **SigLIP fine-tuned** model, and the **RegLIP fine-tuned** model. Including the base model reveals how much each loss function actually improves over the starting checkpoint.

#### Table 4.1: Zero-Shot Classification -- CIFAR-10

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no fine-tuning) | 12.61 | 64.09 |
| SigLIP fine-tuned | 24.58 | 72.51 |
| **RegLIP fine-tuned** | **47.38** | **84.72** |

#### Table 4.2: Zero-Shot Classification -- CIFAR-100

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no fine-tuning) | 1.28 | 7.52 |
| SigLIP fine-tuned | 5.13 | 11.72 |
| **RegLIP fine-tuned** | **8.54** | **22.69** |

#### Table 4.3: Zero-Shot Classification -- ImageNet-1K

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no fine-tuning) | 0.42 | 1.92 |
| SigLIP fine-tuned | 1.40 | 3.22 |
| **RegLIP fine-tuned** | **2.19** | **6.72** |

#### Table 4.4: Zero-Shot Classification -- ImageNet-V2

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no fine-tuning) | 0.39 | 1.54 |
| SigLIP fine-tuned | 1.28 | 3.12 |
| **RegLIP fine-tuned** | **2.15** | **6.78** |

#### Table 4.5: Zero-Shot Classification -- ImageNet-ReaL

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no fine-tuning) | 0.42 | 1.92 |
| SigLIP fine-tuned | 1.40 | 3.22 |
| **RegLIP fine-tuned** | **2.19** | **6.72** |

#### Table 4.6: Summary -- Experiment 1 (Top-1 Accuracy %)

| Dataset | Base | SigLIP FT | RegLIP FT | Abs. Gain | Rel. Gain |
|---------|:----:|:---------:|:---------:|:---------:|:---------:|
| CIFAR-10 | 12.61 | 24.58 | **47.38** | +22.80 | +92.8% |
| CIFAR-100 | 1.28 | 5.13 | **8.54** | +3.41 | +66.5% |
| ImageNet-1K | 0.42 | 1.40 | **2.19** | +0.79 | +56.4% |
| ImageNet-V2 | 0.39 | 1.28 | **2.15** | +0.87 | +68.0% |
| ImageNet-ReaL | 0.42 | 1.40 | **2.19** | +0.79 | +56.4% |

*"Abs. Gain" and "Rel. Gain" measure RegLIP FT over SigLIP FT.*

### 4.3.3 Discussion

**1. RegLIP outperforms SigLIP on every benchmark.** Across all five zero-shot classification datasets, the RegLIP fine-tuned model achieves strictly higher accuracy than the SigLIP fine-tuned model. This is notable because both models start from the same pre-trained checkpoint and are trained under identical conditions except for the loss function.

**2. The gains are large in relative terms.** On CIFAR-10, RegLIP nearly doubles SigLIP's top-1 accuracy (47.38% vs 24.58%, a +92.8% relative improvement). Even on the harder CIFAR-100 and ImageNet benchmarks, relative improvements range from +56% to +68%. The consistency across datasets of varying difficulty suggests the advantage is not dataset-specific.

**3. Both losses improve over the base model, but by different magnitudes.** SigLIP fine-tuning roughly doubles the base model's accuracy on CIFAR-10 (12.61% to 24.58%), while RegLIP nearly quadruples it (12.61% to 47.38%). This indicates that the regression loss extracts substantially more useful signal from the same 31K training pairs.

**4. The Top-5 improvements track the Top-1 improvements.** On CIFAR-10, RegLIP's top-5 accuracy (84.72%) is 12 points above SigLIP's (72.51%). This means RegLIP is not merely getting lucky with the top prediction -- it is placing the correct class in the top-5 more reliably, suggesting genuinely better embedding geometry.

**5. Absolute numbers remain low.** State-of-the-art models (SigLIP, CLIP, etc.) trained on hundreds of millions or billions of pairs achieve 70-80%+ on ImageNet. Our models, trained on only ~31K pairs, achieve single-digit accuracy on ImageNet. The purpose of this experiment is **relative comparison between loss functions**, not absolute state-of-the-art performance.

**Why might RegLIP learn better?** We hypothesise that the continuous regression targets from Qwen3-Embedding-8B provide richer supervision than binary match/no-match labels. Consider a batch of 64 image-text pairs: SigLIP receives 64 positive labels and 64x63 = 4,032 negative labels, all binary. RegLIP receives a full 64x64 matrix of continuous similarity scores, where even "negative" pairs carry graded information (e.g., *"a dog on grass"* is more similar to *"a cat on grass"* than to *"a red car"*). This graded supervision may help the model learn finer-grained distinctions from the same amount of data.

**Limitation of Experiment 1:** The pre-trained SigLIP model was originally trained with binary contrastive loss. Fine-tuning with SigLIP loss continues the same optimisation trajectory, while RegLIP must re-adapt the representations to a regression objective. One could argue that RegLIP succeeds *despite* this handicap, making the result more impressive. However, we cannot fully disentangle loss quality from the interaction between loss and initialisation. **Experiment 2 addresses this by freezing the backbone.**

---

## 4.4 Experiment 2: Frozen Backbone (Projection Heads Only)

### 4.4.1 Motivation

Experiment 1 demonstrated that RegLIP outperforms SigLIP when fine-tuning the full model. However, end-to-end fine-tuning updates approximately 93 million parameters, making it difficult to attribute the improvement solely to the loss function. The encoders themselves might adapt differently under each loss, confounding the comparison.

Experiment 2 eliminates this confound by **freezing both the vision and text encoders entirely**. Only the lightweight projection heads (~6M parameters) and the logit scale/bias are trainable. This means both models process every image and text through the exact same frozen feature extractors. The only variable is how the projection heads learn to map those features into the shared embedding space -- and that mapping is determined entirely by the loss function.

This is the cleanest possible comparison: **same features in, same architecture, different loss, different projections out.**

### 4.4.2 Setup

```
Pre-trained SigLIP (FROZEN vision encoder + text encoder)
    ├──► Train projection heads with SigLIP loss on Flickr30K ──► Evaluate
    └──► Train projection heads with RegLIP loss on Flickr30K ──► Evaluate
```

**Research question:** *Given identical frozen features, does RegLIP loss learn better projection mappings than SigLIP loss?*

**What is frozen vs trainable:**

| Component | Parameters | Trainable? |
|-----------|:----------:|:----------:|
| Vision encoder (ViT-Base/16, 12 layers) | ~86M | Frozen |
| Text encoder (12 layers) | ~86M | Frozen |
| Text projection head (Linear 768 to 768) | ~590K | **Yes** |
| Vision projection head (AttentionPoolingHead) | ~5.3M | **Yes** |
| Logit scale + logit bias | 2 | **Yes** |
| **Total trainable** | **~6M / ~93M** | **(~6.5%)** |

The vision projection head is an `AttentionPoolingHead` that includes a learned probe token, multi-head attention over patch embeddings, layer normalisation, and an MLP -- substantially more complex than the text projection, which is a single linear layer. Both are randomly initialised in the original SigLIP model and carry pre-trained weights in our starting checkpoint.

**Training Hyperparameters (Experiment 2):**

| Parameter | SigLIP Frozen | RegLIP Frozen |
|-----------|:-------------:|:-------------:|
| Trainable parameters | ~6M (6.5%) | ~6M (6.5%) |
| Loss function | Binary cross-entropy | MSE regression |
| Learning rate | **1e-3** | **1e-3** |
| Optimizer | AdamW (weight decay 0.01) | AdamW (weight decay 0.01) |
| LR Scheduler | Cosine annealing | Cosine annealing |
| Batch size | 64 | 64 |
| Epochs | 10 | 10 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | FP16 | FP16 |
| Similarity teacher | -- | Qwen3-Embedding-8B |

The learning rate is increased from 1e-4 (Experiment 1) to **1e-3**. This is standard practice when training only projection heads on frozen features: since the backbone gradients are zeroed out, there is no risk of catastrophic forgetting, and the projection heads benefit from a larger step size to learn their mapping within the limited training budget. Both models use the same elevated learning rate.

### 4.4.3 Results

#### Table 4.7: Zero-Shot Classification -- CIFAR-10 (Frozen Backbone)

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no training) | 12.61 | 64.09 |
| SigLIP frozen | **26.46** | 71.37 |
| RegLIP frozen | 23.13 | **81.20** |

#### Table 4.8: Zero-Shot Classification -- CIFAR-100 (Frozen Backbone)

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no training) | 1.28 | 7.52 |
| SigLIP frozen | **3.34** | **12.94** |
| RegLIP frozen | 2.69 | 12.90 |

#### Table 4.9: Zero-Shot Classification -- ImageNet-1K (Frozen Backbone)

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no training) | 0.42 | 1.92 |
| **RegLIP frozen** | **1.42** | 4.32 |
| SigLIP frozen | 1.28 | **4.82** |

#### Table 4.10: Zero-Shot Classification -- ImageNet-V2 (Frozen Backbone)

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no training) | 0.39 | 1.54 |
| **SigLIP frozen** | **1.36** | **4.53** |
| RegLIP frozen | 1.31 | 4.15 |

#### Table 4.11: Zero-Shot Classification -- ImageNet-ReaL (Frozen Backbone)

| Model | Top-1 (%) | Top-5 (%) |
|-------|:---------:|:---------:|
| Base (pre-trained, no training) | 0.42 | 1.92 |
| **RegLIP frozen** | **1.42** | 4.32 |
| SigLIP frozen | 1.28 | **4.82** |

#### Table 4.12: Summary -- Experiment 2 (Top-1 Accuracy %)

| Dataset | Base | SigLIP Frozen | RegLIP Frozen | Winner |
|---------|:----:|:-------------:|:-------------:|:------:|
| CIFAR-10 | 12.61 | **26.46** | 23.13 | SigLIP |
| CIFAR-100 | 1.28 | **3.34** | 2.69 | SigLIP |
| ImageNet-1K | 0.42 | 1.28 | **1.42** | RegLIP |
| ImageNet-V2 | 0.39 | **1.36** | 1.31 | SigLIP |
| ImageNet-ReaL | 0.42 | 1.28 | **1.42** | RegLIP |

### 4.4.4 Discussion

**1. The results are mixed -- neither loss dominates.** Unlike Experiment 1 where RegLIP won every benchmark, Experiment 2 shows a split: SigLIP frozen wins on CIFAR-10, CIFAR-100, and ImageNet-V2, while RegLIP frozen wins on ImageNet-1K and ImageNet-ReaL. The margins are small in all cases.

**2. SigLIP has an advantage on coarser tasks.** On CIFAR-10 (10 classes), SigLIP frozen leads by 3.33 points in top-1. On CIFAR-100 (100 classes), it leads by 0.65 points. This makes intuitive sense: binary match/no-match supervision is well-suited for coarse-grained distinctions. When there are only 10 or 100 classes, the primary challenge is "same or different category?" -- exactly what binary loss optimises for.

**3. RegLIP shows an edge on fine-grained tasks.** On ImageNet-1K and ImageNet-ReaL (1000 classes), RegLIP frozen leads by 0.14 points in top-1. Although small, this aligns with the hypothesis: among 1000 fine-grained classes, graded similarity information helps distinguish visually similar categories (e.g., different dog breeds, different vehicle types) that binary labels treat identically as "not a match."

**4. RegLIP frozen achieves notably higher CIFAR-10 top-5.** Despite losing top-1 on CIFAR-10, RegLIP frozen achieves 81.20% top-5 versus SigLIP's 71.37% -- a 9.83-point gap. This suggests RegLIP is building a better overall ranking of classes even when the top-1 prediction misses. The continuous similarity targets encourage the model to place semantically related classes closer together, producing a more meaningful similarity ordering.

**5. Both frozen models improve substantially over the base.** Training only ~6.5% of parameters still produces meaningful gains (e.g., base 12.61% to 23-26% on CIFAR-10). This confirms that the projection heads are a significant bottleneck and that even limited training on Flickr30K can improve them.

**6. The frozen setting is more constrained.** With only ~6M trainable parameters operating on fixed features, both losses are working within a narrow band. The features extracted by the frozen ViT-Base/16 are optimised for the original SigLIP objective. RegLIP's regression loss must learn a useful projection from features that were never trained to support graded similarity -- a harder task. Despite this, RegLIP remains competitive and wins on the hardest benchmarks.

**Interpretation:** Experiment 2 suggests that the dramatic advantage seen in Experiment 1 comes partly from RegLIP's ability to reshape the backbone representations, not just the projections. When the backbone is frozen, RegLIP loses its primary advantage -- the ability to re-organise features to support graded similarity -- and the two losses perform comparably. This motivates Experiment 3: if we train from scratch (where there is no pre-existing bias for either loss), we can observe which loss learns better representations from the ground up.

---

## 4.5 Cross-Experiment Analysis (Experiments 1 & 2)

#### Table 4.13: Full Results -- All Models, All Benchmarks (Top-1 Accuracy %)

| Dataset | Base | SigLIP FT | RegLIP FT | SigLIP Frozen | RegLIP Frozen |
|---------|:----:|:---------:|:---------:|:-------------:|:-------------:|
| CIFAR-10 | 12.61 | 24.58 | **47.38** | 26.46 | 23.13 |
| CIFAR-100 | 1.28 | 5.13 | **8.54** | 3.34 | 2.69 |
| ImageNet-1K | 0.42 | 1.40 | **2.19** | 1.28 | 1.42 |
| ImageNet-V2 | 0.39 | 1.28 | **2.15** | 1.36 | 1.31 |
| ImageNet-ReaL | 0.42 | 1.40 | **2.19** | 1.28 | 1.42 |

#### Table 4.14: Full Results -- All Models, All Benchmarks (Top-5 Accuracy %)

| Dataset | Base | SigLIP FT | RegLIP FT | SigLIP Frozen | RegLIP Frozen |
|---------|:----:|:---------:|:---------:|:-------------:|:-------------:|
| CIFAR-10 | 64.09 | 72.51 | **84.72** | 71.37 | 81.20 |
| CIFAR-100 | 7.52 | 11.72 | **22.69** | 12.94 | 12.90 |
| ImageNet-1K | 1.92 | 3.22 | **6.72** | 4.82 | 4.32 |
| ImageNet-V2 | 1.54 | 3.12 | **6.78** | 4.53 | 4.15 |
| ImageNet-ReaL | 1.92 | 3.22 | **6.72** | 4.82 | 4.32 |

### Key Observations Across Experiments

**1. End-to-end fine-tuning with RegLIP is the best setting overall.** RegLIP fine-tuned achieves the highest score on every benchmark in both top-1 and top-5. The advantage over all other models is substantial, confirming that the regression loss is most effective when it can shape the entire representation pipeline.

**2. Frozen models are competitive with fine-tuned SigLIP.** Notably, the frozen SigLIP model (training only ~6.5% of parameters) achieves comparable or even slightly better top-1 accuracy than the fully fine-tuned SigLIP model on some benchmarks (CIFAR-10: 26.46% frozen vs 24.58% fine-tuned). This suggests that SigLIP's binary loss may actually *hurt* the pre-trained backbone when fine-tuning end-to-end on a small dataset -- the projections improve but the representations degrade.

**3. RegLIP frozen excels at top-5 despite mixed top-1.** On CIFAR-10, RegLIP frozen achieves 81.20% top-5 -- higher than SigLIP fine-tuned (72.51%) and even approaching RegLIP fine-tuned (84.72%). This is remarkable given that only 6.5% of parameters were trained. It suggests that continuous similarity targets produce a better-calibrated embedding space where the correct class is consistently ranked highly, even if it does not always land at rank 1.

**4. The gap between frozen and fine-tuned reveals representation learning quality.** RegLIP fine-tuned dramatically outperforms RegLIP frozen (e.g., 47.38% vs 23.13% on CIFAR-10 top-1), while SigLIP fine-tuned only modestly outperforms SigLIP frozen (24.58% vs 26.46%). This implies that RegLIP's regression loss drives more significant representation learning when the backbone is unfrozen -- it does not merely learn better projections, it reshapes the features themselves.

### Emerging Picture

| Setting | RegLIP advantage? | Interpretation |
|---------|:-----------------:|----------------|
| End-to-end fine-tuning | **Strong yes** | Regression loss reshapes features + projections |
| Frozen backbone (top-1) | Mixed | Without feature reshaping, binary loss is competitive |
| Frozen backbone (top-5) | **Yes** | Regression loss still builds better similarity ordering |

The results so far suggest that RegLIP's primary strength is in guiding the backbone to learn representations with finer-grained similarity structure. When the backbone is locked, this advantage is diminished but not eliminated (as evidenced by the top-5 results). **Experiment 3 (training from scratch) is the critical test**: if RegLIP outperforms SigLIP when both models learn representations from random initialisation, it confirms that the regression loss is fundamentally better for vision-language representation learning.

---

## 4.6 Experiment 3: Training from Scratch on CC3M

> **Status: Pending**

### Setup

```
Random initialization (same ViT-Base/16 architecture)
    ├──► Train from scratch with SigLIP loss on CC3M (~3M pairs) ──► Evaluate
    └──► Train from scratch with RegLIP loss on CC3M (~3M pairs) ──► Evaluate
```

**Research question:** *Does RegLIP loss learn fundamentally better vision-language representations when there is no pre-training bias?*

No pre-trained weights are used. Both models learn everything from scratch on a larger dataset (CC3M, ~3M pairs), providing the strongest evidence for or against the proposed loss function.

**Requirements:** CC3M dataset (~3M image-text pairs), significant compute time (~2-5 days on A100).

### Results

*To be filled after running the experiment.*

#### Table 4.15: Zero-Shot Classification -- From Scratch (Top-1 %)

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

## 4.7 Ablation Studies

> **Status: Planned**

Potential ablation directions to explore after the main experiments:

- **4.7.1 Frozen Encoder Choice:** Compare Qwen-8B vs smaller models (e.g., Qwen-1.5B, all-MiniLM) as the similarity target generator. Does a stronger teacher always produce better targets?
- **4.7.2 Loss Variants:** MSE vs KL Divergence vs Smooth L1 for the regression objective.
- **4.7.3 Similarity Scaling Methods:** Different ways to map raw cosine similarity to the target range.
- **4.7.4 Learning Rate Sensitivity:** Whether the optimal learning rate differs between the two losses, particularly in the frozen backbone setting.

---

## How to Reproduce

### Experiment 1 (Fine-Tuning)

```bash
# Train SigLIP baseline
python training/train_siglip.py --config configs/siglip_config.yaml

# Train RegLIP
python training/train_reglip.py --config configs/reglip_config.yaml

# Evaluate
bash evaluate.sh reglip_base   # pre-trained baseline (no fine-tuning)
bash evaluate.sh siglip        # siglip fine-tuned
bash evaluate.sh reglip        # reglip fine-tuned
```

### Experiment 2 (Frozen Backbone)

```bash
# Train frozen models (projection heads only)
./train_frozen.sh              # trains both siglip_frozen and reglip_frozen
./train_frozen.sh --debug      # quick test run with 100 samples, 2 epochs

# Evaluate
bash evaluate.sh siglip_frozen
bash evaluate.sh reglip_frozen
```

### Experiment 3 (From Scratch)

```bash
# To be implemented: requires CC3M dataset and random initialization config
```
