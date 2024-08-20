<<<<<<< HEAD
# RegLIP: Regression-Based Contrastive Learning for Vision-Language Models

A research project exploring regression-based contrastive learning as an alternative to binary contrastive learning in vision-language models.

## Research Motivation

### Standard CLIP/SigLIP Approach

In traditional contrastive learning (CLIP, SigLIP), the training uses **binary labels**:
- **1** for matching image-text pairs (diagonal of the similarity matrix)
- **0** for all non-matching pairs (off-diagonal)

```
Labels Matrix (Binary):
        Text_0  Text_1  Text_2  Text_3
Image_0   1       0       0       0
Image_1   0       1       0       0
Image_2   0       0       1       0
Image_3   0       0       0       1
```

**Problem**: This approach treats all non-matching pairs equally, ignoring semantic relationships. For example:
- "A dog playing in the park" and "A puppy running on grass" are treated as completely unrelated (0)
- Even though they're semantically very similar

### RegLIP Approach (This Research)

Instead of binary labels, RegLIP uses **continuous similarity scores** as targets:

```
Similarity Targets (Continuous):
        Text_0  Text_1  Text_2  Text_3
Image_0  1.00    0.85    0.32    0.15
Image_1  0.85    1.00    0.28    0.12
Image_2  0.32    0.28    1.00    0.45
Image_3  0.15    0.12    0.45    1.00
```

**Key Idea**: Use a frozen text encoder (Qwen-8B embeddings) to compute text-text similarities, then use these as soft labels to train the vision-language model.

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RegLIP Training                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │   Images     │         │   Captions   │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │   Vision     │         │    Text      │                      │
│  │   Encoder    │         │   Encoder    │                      │
│  │ (trainable)  │         │ (trainable)  │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         ▼                        ▼                               │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ Image Embeds │         │ Text Embeds  │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         └────────┬───────────────┘                               │
│                  ▼                                               │
│         ┌──────────────────┐                                    │
│         │ Image-Text       │                                    │
│         │ Similarity (P)   │◄──── Predictions                   │
│         └────────┬─────────┘                                    │
│                  │                                               │
│                  ▼                                               │
│         ┌──────────────────┐      ┌──────────────────┐          │
│         │   MSE Loss       │◄─────│ Text-Text        │          │
│         │                  │      │ Similarity (S)   │◄── Targets│
│         └──────────────────┘      └────────┬─────────┘          │
│                                            │                     │
│                                   ┌────────┴─────────┐          │
│                                   │  Frozen Qwen-8B  │          │
│                                   │  Text Encoder    │          │
│                                   └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Training Flow

1. **Input**: Batch of (Image, Caption) pairs
2. **Targets Generation**: 
   - Pass all captions through frozen Qwen-8B encoder
   - Compute text-text cosine similarity matrix
   - Scale to [0, 1] range
3. **Predictions**:
   - Pass images through trainable vision encoder
   - Pass captions through trainable text encoder
   - Compute image-text similarity matrix
4. **Loss**: MSE between predicted similarities and target similarities

### Key Assumption

The model learns: `similarity(Image_i, Text_j) ≈ similarity(Text_i, Text_j)`

**Interpretation**: If two captions describe semantically similar scenes, the model should recognize that an image matching one caption should also have some relationship with the other caption.

## Project Structure

```
RegLIP/
├── configs/
│   ├── reglip_config.yaml    # RegLIP training configuration
│   └── siglip_config.yaml    # SigLIP baseline configuration
├── data/
│   ├── dataset.py            # Flickr30K dataset implementation
│   ├── preprocessing.py      # Image/text processing utilities
│   └── flickr30k/            # Dataset directory
│       ├── Images/           # Image files
│       └── captions.txt      # Image-caption pairs
├── reglip/
│   ├── model.py              # RegLIP model implementation
│   ├── config.py             # Model configuration
│   ├── text.py               # Text encoder
│   ├── vision.py             # Vision encoder
│   ├── embedding_utils.py    # Qwen embedding client
│   └── utils.py              # Model utilities
├── training/
│   ├── trainer.py            # Training loop (RegLIP & SigLIP)
│   ├── train_reglip.py       # RegLIP training script
│   ├── train_siglip.py       # SigLIP training script
│   └── utils.py              # Training utilities
├── evaluation/
│   └── metrics.py            # Evaluation metrics
├── checkpoints/              # Model checkpoints
├── runs/                     # TensorBoard logs
├── logs/                     # Training logs
└── train.sh                  # Training launch script
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RegLIP

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### RegLIP Configuration (`configs/reglip_config.yaml`)

```yaml
model:
  name: "reglip"
  pretrained_model: "google/siglip-base-patch16-224"
  reglip_config:
    frozen_text_encoder: "qwen"
    qwen_api_url: "http://your-qwen-api:6010"
    qwen_model: "Qwen/Qwen3-Embedding-8B"

training:
  batch_size: 128
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
  max_epochs: 10

logging:
  use_tensorboard: true
  tensorboard_log_dir: "./runs"
  use_wandb: false
```

## Training

### Train RegLIP

```bash
# Using the training script
bash train.sh

# Or directly
python training/train_reglip.py --config configs/reglip_config.yaml
```

### Train SigLIP (Baseline)

```bash
python training/train_siglip.py --config configs/siglip_config.yaml
```

### Monitor Training

```bash
# TensorBoard (logs are inside the checkpoint directory)
tensorboard --logdir ./checkpoints/reglip/tensorboard   # RegLIP
tensorboard --logdir ./checkpoints/siglip/tensorboard   # SigLIP

# Open http://localhost:6006 in your browser
```

## Implementation Details

### Loss Functions

#### SigLIP Binary Loss (Baseline)
```python
# Binary targets: diagonal = 1, off-diagonal = -1
eye = torch.eye(batch_size)
targets = -torch.ones_like(logits) + 2 * eye

# Log-sigmoid loss
loss = -torch.sum(F.logsigmoid(targets * logits), dim=-1).mean()
```

#### RegLIP Regression Loss (Proposed)
```python
# Continuous targets from frozen text encoder (Qwen-8B)
similarity_targets = frozen_encoder.compute_text_similarity(captions)

# Raw cosine similarity mapped from [-1, 1] to [0, 1]
cosine_sim = torch.matmul(text_embeds, image_embeds.t())
predicted = (cosine_sim + 1) / 2

# MSE loss
loss = F.mse_loss(predicted, similarity_targets)
```

### Similarity Target Generation

```python
def generate_similarity_targets(texts):
    # 1. Get embeddings from frozen Qwen encoder
    embeddings = qwen_encoder.get_embeddings(texts)
    
    # 2. Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 3. Compute cosine similarity matrix
    similarity = torch.matmul(embeddings, embeddings.t())
    
    # 4. Scale from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2
    
    return similarity
```

## Potential Advantages

1. **Richer Supervision**: Continuous targets preserve semantic relationships between all captions in a batch

2. **Addresses False Negatives**: Similar captions get high similarity scores instead of being treated as negatives

3. **Knowledge Transfer**: Distills knowledge from powerful language model (Qwen-8B) into lighter CLIP encoder

4. **Smoother Optimization**: Continuous targets may provide more stable gradients

## Evaluation Metrics

- **Image-to-Text Retrieval**: R@1, R@5, R@10
- **Text-to-Image Retrieval**: R@1, R@5, R@10
- **Zero-Shot Classification**: Accuracy on standard benchmarks

## Checkpoints

Each model's outputs are organized under `./checkpoints/<model>/`:

```
checkpoints/reglip/
├── logs/                  # Training text logs
├── tensorboard/           # TensorBoard event files
├── epoch_1.pth            # Checkpoint after epoch 1
├── epoch_2.pth            # Checkpoint after epoch 2
├── ...
├── epoch_10.pth           # Checkpoint after epoch 10
└── best_model.pth         # Best model (lowest validation loss)
```

## TensorBoard Metrics

TensorBoard logs are stored at `./checkpoints/<model>/tensorboard/`. The following metrics are logged:
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/gradient_norm`: Gradient norm
- `train/similarity_mean`: Mean of similarity targets (RegLIP only)
- `train/similarity_std`: Std of similarity targets (RegLIP only)
- `val/val_loss`: Validation loss

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{reglip2024,
  title={RegLIP: Regression-Based Contrastive Learning for Vision-Language Models},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

[Your License Here]

## Acknowledgments

- Based on [SigLIP](https://arxiv.org/abs/2303.15343) architecture
- Uses [Qwen](https://github.com/QwenLM/Qwen) embeddings for similarity targets
- Dataset: [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)
=======
# Reglip implementation
>>>>>>> 5b5a165 (edit: edit readme file)
