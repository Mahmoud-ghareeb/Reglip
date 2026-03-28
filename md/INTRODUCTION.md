# Chapter 1: Introduction

## 1.1 The Rise of Vision-Language Models

The ability to understand images and text jointly has become one of the central goals of modern artificial intelligence. Vision-language models (VLMs) learn a shared embedding space where images and their textual descriptions are mapped to nearby points, enabling tasks such as zero-shot image classification, image-text retrieval, and visual question answering — all without task-specific training data.

The dominant paradigm for training such models is **contrastive learning**. CLIP (Radford et al., 2021) demonstrated that a simple dual-encoder architecture — a vision encoder and a text encoder trained to maximise the similarity of matching image-text pairs and minimise the similarity of non-matching ones — can learn remarkably powerful representations when scaled to hundreds of millions of image-text pairs from the internet. This approach has since been extended and refined by models such as ALIGN (Jia et al., 2021), SigLIP (Zhai et al., 2023), and OpenCLIP (Cherti et al., 2023), each improving on the training objective, architecture, or data pipeline.

Despite their success, all of these models share a fundamental design choice: they treat every image-text pair as either a **match** or a **non-match** — a binary label of 1 or 0. This thesis questions whether this binary supervision is a bottleneck, and proposes an alternative.

## 1.2 The Problem: Binary Labels Discard Semantic Structure

Consider a training batch containing the following three captions:

- **Caption A:** *"A golden retriever playing fetch on a sunny beach"*
- **Caption B:** *"A dog running on sand near the ocean"*
- **Caption C:** *"A red sports car parked in a garage"*

In standard contrastive learning, each caption is paired with its corresponding image. All other pairs in the batch are treated as negatives and assigned a label of 0. This means the model is told that Image A is equally unrelated to Caption B as it is to Caption C — even though captions A and B describe nearly identical scenes, while caption C describes something entirely different.

This is the **false negative problem** at its core, but it extends beyond simple false negatives. Even among pairs that are genuinely not exact matches, there exists a rich spectrum of semantic relatedness:

- *"A dog on a beach"* and *"A puppy near the sea"* — highly related
- *"A dog on a beach"* and *"Children building a sandcastle"* — moderately related (shared beach context)
- *"A dog on a beach"* and *"A circuit board under a microscope"* — completely unrelated

Binary labels collapse this spectrum into a single bit of information. Every pair that is not an exact match receives the same supervision signal, regardless of how semantically close or distant the pair actually is. In a batch of 64 pairs, the model receives 64 positive labels and 4,032 negative labels — and all 4,032 negatives are treated identically.

**The information loss is substantial.** The full pairwise similarity structure of a batch of 64 captions can be described by a 64 × 64 matrix of continuous values — 4,096 numbers carrying graded semantic information. Binary contrastive learning reduces this to 4,096 binary values. The question this thesis asks is: **can we recover some of that lost information, and does it help?**

## 1.3 The Proposed Solution: RegLIP

We propose **RegLIP** (Regression-based Language-Image Pre-training), a modification to the contrastive learning framework that replaces binary match/no-match labels with **continuous similarity targets**.

The key idea is simple. Instead of telling the model "this image matches this text (1) and not those texts (0)," we tell it "this image should have similarity 1.0 with this text, 0.85 with that text, 0.32 with that other text, and 0.15 with that one." The continuous targets capture the full spectrum of semantic relatedness within each training batch.

### Where Do the Continuous Targets Come From?

We use a **frozen teacher model** — specifically, Qwen3-Embedding-8B (Alibaba, 2025), a large text embedding model with 8 billion parameters — to generate pairwise similarity scores between all captions in a training batch. For each batch:

1. All captions are passed through the frozen Qwen3-Embedding-8B encoder to obtain text embeddings.
2. Cosine similarity is computed between every pair of caption embeddings, producing a similarity matrix.
3. The similarity values are scaled from [-1, 1] to [0, 1] to serve as regression targets.

The trainable student model (a standard SigLIP-sized dual encoder with ~93M parameters) then learns to predict these continuous similarity targets using Mean Squared Error (MSE) regression, instead of the binary cross-entropy loss used by SigLIP.

### What Changes and What Stays the Same

It is important to emphasise how minimal the modification is:

| Component | SigLIP (Baseline) | RegLIP (Proposed) |
|-----------|:--:|:--:|
| Vision encoder | ViT-Base/16 | ViT-Base/16 (identical) |
| Text encoder | Transformer, 12 layers | Transformer, 12 layers (identical) |
| Projection heads | Same | Same (identical) |
| Training data | Flickr30K | Flickr30K (identical) |
| Forward pass | Same | Same (identical) |
| **Loss function** | **Binary cross-entropy** | **MSE regression** |
| **Supervision signal** | **Binary: 0 or 1** | **Continuous: [0, 1]** |
| Similarity teacher | None | Qwen3-Embedding-8B (frozen, not part of the model) |

The architecture, data, and training recipe are unchanged. The only difference is the loss function and the source of supervision labels. This makes the comparison clean: any difference in downstream performance must be attributable to the quality of supervision.

### The Core Assumption

RegLIP is built on a single assumption: **if two captions are semantically similar, then an image matching one caption should also be somewhat similar to the other caption.** Formally, for caption *i* paired with image *i*, and any other caption *j* in the batch:

> similarity(Image_i, Text_j) ≈ similarity(Text_i, Text_j)

This transfers text-text semantic structure (as measured by the teacher model) into the image-text embedding space. The student model learns not just *which* pairs match, but *how much* any pair is related.

## 1.4 Why This Matters

### 1.4.1 Richer Supervision from the Same Data

Vision-language datasets are expensive to collect and curate. The largest models (CLIP, SigLIP) are trained on hundreds of millions to billions of image-text pairs, often scraped from the internet with noisy correspondence. In settings where data is limited — domain-specific applications, low-resource languages, academic research budgets — extracting more learning signal from each batch is valuable.

RegLIP achieves this by leveraging a strong text encoder to generate dense supervision. A batch of 64 pairs yields 64 binary labels under SigLIP, but 4,096 continuous similarity values under RegLIP. Each training step carries more information, which may be especially impactful in low-data regimes.

### 1.4.2 Addressing the False Negative Problem

False negatives are a well-known problem in contrastive learning. When a batch contains semantically similar but non-identical pairs, binary labels incorrectly push their representations apart. Prior work has addressed this with larger batch sizes (to reduce collision probability), hard negative mining, or modified sampling strategies. RegLIP addresses it at the label level: similar captions automatically receive high similarity targets, so the model is never told to push genuinely related pairs apart.

### 1.4.3 Knowledge Distillation into Lightweight Models

RegLIP can be viewed as a form of knowledge distillation. The frozen Qwen3-Embedding-8B teacher (8 billion parameters) encodes rich semantic knowledge about language. By using its similarity judgements as targets, we transfer some of that knowledge into a much smaller student model (~93M parameters). The student learns to organise its embedding space in a way that reflects the teacher's understanding of text similarity, without needing to be as large.

### 1.4.4 A Simple, Architecture-Agnostic Modification

Unlike many improvements in vision-language learning that require architectural changes (new attention mechanisms, additional fusion layers, memory banks), RegLIP modifies only the loss function. It can be applied to any contrastive learning framework — CLIP, SigLIP, OpenCLIP, or future architectures — by simply swapping the loss and providing a similarity teacher. This makes it easy to adopt and test.

## 1.5 Research Questions

This thesis investigates the following questions:

1. **Can continuous similarity targets improve fine-tuning of pre-trained vision-language models?** Starting from the same pre-trained SigLIP checkpoint, does fine-tuning with RegLIP loss produce better downstream performance than fine-tuning with SigLIP loss?

2. **Does the advantage come from the loss function or from representation learning?** When the backbone encoders are frozen and only projection heads are trained, does RegLIP still outperform SigLIP? This isolates the loss function's effect on projection learning from its effect on feature learning.

3. **Does RegLIP learn fundamentally better representations when trained from scratch?** Without any pre-training bias, does the regression loss produce a better embedding space than the binary loss? This is the strongest test of the hypothesis.

## 1.6 Contributions

This thesis makes the following contributions:

1. **RegLIP: a regression-based contrastive loss for vision-language models.** We propose replacing binary match/no-match labels with continuous similarity targets from a frozen teacher encoder, using MSE regression instead of binary cross-entropy. The modification is minimal (loss function only) and architecture-agnostic.

2. **A controlled experimental comparison.** We design three experiments — fine-tuning, frozen backbone, and training from scratch — that progressively isolate the loss function's contribution. Each experiment uses the same architecture, data, and hyperparameters, varying only the loss.

3. **Empirical evidence that continuous targets improve fine-tuning.** On five zero-shot classification benchmarks (CIFAR-10, CIFAR-100, ImageNet-1K, ImageNet-V2, ImageNet-ReaL), RegLIP fine-tuned on only 31K image-text pairs outperforms SigLIP by +56% to +93% in relative terms. On CIFAR-10, RegLIP nearly doubles SigLIP's accuracy (47.4% vs 24.6%).

4. **Analysis of where the advantage originates.** Frozen-backbone experiments reveal that RegLIP's primary strength lies in reshaping backbone representations, not merely learning better projections. When the backbone is frozen, the two losses perform comparably on top-1 accuracy, but RegLIP produces superior top-5 rankings — suggesting a better-calibrated embedding geometry.

5. **A complete, reproducible codebase.** All code, configurations, training scripts, and evaluation pipelines are provided, enabling full reproduction of every result reported in this thesis.

## 1.7 Thesis Outline

The remainder of this thesis is organised as follows:

**Chapter 2: Background and Related Work.** We review the foundations of contrastive learning for vision-language models, covering CLIP, SigLIP, and their variants. We discuss the false negative problem, soft-label training, knowledge distillation, and prior work on improving contrastive objectives. We position RegLIP within this landscape and identify the gap it addresses.

**Chapter 3: Methodology.** We present the RegLIP model in detail: the architecture (inherited from SigLIP), the regression loss function, the similarity target generation pipeline using Qwen3-Embedding-8B, and the design decisions behind each component. We also describe the three experimental settings and the evaluation protocol.

**Chapter 4: Experiments and Results.** We report results from all three experiments across five zero-shot classification benchmarks. We provide detailed analysis of when and why RegLIP outperforms SigLIP, including cross-experiment comparisons that reveal the source of the advantage.

**Chapter 5: Conclusion.** We summarise our findings, discuss limitations (dataset scale, absolute performance, teacher model dependency), and outline directions for future work including scaling to larger datasets and exploring alternative teacher models.
