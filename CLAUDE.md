# CLAUDE.md — RegLIP Project

## What is RegLIP?

RegLIP (Regression-based Language-Image Pre-training) enhances SigLIP by replacing binary contrastive loss with MSE regression against continuous similarity targets from a frozen teacher encoder. This addresses false negatives and provides richer supervision than 0/1 labels.

**Core idea**: A frozen text embedding teacher (e.g. Qwen3-8B) generates soft similarity scores for each caption pair in a batch. The student (SigLIP-base, 93M params) learns to predict these continuous targets instead of binary match/no-match.

## Project structure

```
reglip/                  # Main package
  config.py              # RegLIPConfig, RegLIPTextConfig, RegLIPVisionConfig
  model.py               # RegLIPModel — forward pass, losses, frozen encoder interface
  text.py                # Text encoder (transformer)
  vision.py              # Vision encoder (ViT)
  utils.py               # SigLIP loading, setup_frozen_text_encoder(), freeze_backbone()
  embedding_utils.py     # Legacy re-exports (backward compat for QwenEmbeddingClient)
  embeddings/            # Pluggable teacher embedding models
    base.py              # BaseEmbeddingModel ABC
    qwen_api.py          # QwenAPIEmbedding — remote vLLM API client
    omni_embed.py        # OmniEmbedEmbedding — local Tevatron/OmniEmbed-v0.1
    __init__.py          # EMBEDDING_REGISTRY + create_embedding_model() factory

training/                # Training infrastructure
  trainer.py             # BaseTrainer, SigLIPTrainer, RegLIPTrainer
  train_reglip.py        # RegLIP training entry point
  train_siglip.py        # SigLIP baseline training entry point
  train_rsicd.py         # RSICD domain training (supports both losses)
  utils.py               # set_seed, load_config, create_optimizer_and_scheduler, etc.

evaluation/              # Evaluation framework
  evaluator.py           # Unified Evaluator class
  metrics.py             # compute_retrieval_metrics, compute_accuracy
  datasets/              # Dataset classes (flickr30k, cifar, imagenet, coco, rsicd, patternnet)
  tasks/                 # RetrievalTask, ZeroShotTask

scripts/                 # Utility scripts
  run_evaluation.py      # Main evaluation CLI
  download_imagenet.py
  validate_dataset.py

configs/                 # YAML configs — one per experiment variant
  reglip_config.yaml               # RegLIP + Flickr30K
  siglip_config.yaml               # SigLIP baseline + Flickr30K
  reglip_frozen_config.yaml        # Frozen backbone variant
  siglip_frozen_config.yaml
  reglip_rsicd_config.yaml         # RSICD with Qwen teacher
  siglip_rsicd_config.yaml
  reglip_rsicd_omniembed_config.yaml  # RSICD with OmniEmbed teacher
  eval_config.yaml                 # Evaluation settings
```

## How to run

### Training

```bash
# RegLIP on Flickr30K
python training/train_reglip.py --config configs/reglip_config.yaml --data_root ./data/flickr30k

# SigLIP baseline
python training/train_siglip.py --config configs/siglip_config.yaml --data_root ./data/flickr30k

# RSICD domain
python training/train_rsicd.py --config configs/reglip_rsicd_config.yaml --data_root ./data/rsicd

# Shell launchers
./train.sh reglip          # or siglip, both
./train_rsicd.sh reglip    # or siglip, both, eval, all

# Debug mode (100 samples, 2 epochs, no wandb)
python training/train_reglip.py --config configs/reglip_config.yaml --debug
```

CLI args common to all training scripts: `--config`, `--data_root`, `--seed` (default 42), `--debug`.

### Evaluation

```bash
./evaluate.sh reglip       # or siglip, compare, retrieval, zero_shot, imagenet, etc.

python scripts/run_evaluation.py \
  --checkpoint ./checkpoints/reglip/best_model.pth \
  --model_type reglip \
  --dataset cifar10 flickr30k \
  --task zero_shot retrieval
```

## Model architecture

- **Base**: `google/siglip-base-patch16-224` (loaded via `load_from_transformers_siglip()`)
- **Text encoder**: 12-layer transformer, hidden=768, vocab=32000, max_seq=64
- **Vision encoder**: 12-layer ViT, hidden=768, patch=16, image=224 (196 patches)
- **Projection dim**: 768
- **Parameters**: ~93M student (SigLIP-base)

### Loss functions

1. **RegLIP regression loss** (`regression_contrastive_loss`): MSE between predicted cosine similarity (mapped to [0,1]) and frozen teacher targets
2. **SigLIP binary loss** (`binary_contrastive_loss`): log-sigmoid loss with +1/-1 targets (diagonal positive, off-diagonal negative)

The regression loss uses raw cosine similarity (no logit_scale/bias) to isolate the soft-label effect.

## Embedding system (teacher models)

Teacher models live in `reglip/embeddings/`. All implement `BaseEmbeddingModel`:

```python
class BaseEmbeddingModel(ABC):
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray: ...
    @property
    def embedding_dim(self) -> int: ...
    @property
    def name(self) -> str: ...
```

### Registry

| Key          | Class                | Dim  | Type         |
|-------------|---------------------|------|--------------|
| `qwen_api`   | `QwenAPIEmbedding`   | 4096 | Remote API   |
| `omni_embed` | `OmniEmbedEmbedding` | 3584 | Remote API   |

### Adding a new embedding model

1. Create `reglip/embeddings/your_model.py` extending `BaseEmbeddingModel`
2. Add to `EMBEDDING_REGISTRY` in `reglip/embeddings/__init__.py`
3. In config YAML:
   ```yaml
   reglip_config:
     frozen_text_encoder: "your_key"
     encoder_kwargs:
       param1: value1
   ```

### Config patterns

**Qwen API (legacy keys still supported)**:
```yaml
reglip_config:
  frozen_text_encoder: "qwen"      # or "qwen_api"
  qwen_api_url: "http://212.41.29.82:6010"
  qwen_model: "Qwen/Qwen3-Embedding-8B"
```

**OmniEmbed (API)**:
```yaml
reglip_config:
  frozen_text_encoder: "omni_embed"
  encoder_kwargs:
    base_url: "http://212.41.29.82:6020"
    model: "Tevatron/OmniEmbed-v0.1"
```

## Key APIs

```python
# Factory
from reglip.embeddings import create_embedding_model
encoder = create_embedding_model("qwen_api", base_url="...", model="...")

# Set on model
model.set_frozen_text_encoder(encoder)

# During training forward pass
targets = model.generate_similarity_targets(captions)  # [batch, batch] in [0,1]
```

`setup_frozen_text_encoder(model, encoder_name, **kwargs)` in `reglip/utils.py` handles this from config values.

## Data

- **Flickr30K**: 31K images, 5 captions each. Stored in `data/flickr30k/`
- **RSICD**: Remote sensing image captioning. Stored in `data/rsicd/` with `train.csv`, `test.csv`, `valid.csv`
- **CIFAR-10/100**: Downloaded automatically via torchvision
- **ImageNet variants**: ImageNet-1K, V2, ReaL, ObjectNet. Set `IMAGENET_ROOT` env var
- **PatternNet**: Remote sensing classification

All eval datasets extend `BaseEvalDataset` in `evaluation/datasets/base.py`.

## Config structure (YAML)

```yaml
model:
  pretrained_model: "google/siglip-base-patch16-224"
  load_pretrained: true
  freeze_backbone: false        # true = only train projection heads
  reglip_config:
    frozen_text_encoder: "qwen_api"
    encoder_kwargs: {}          # passed to create_embedding_model()
    similarity_loss_weight: 1.0
    logit_scale_init_value: 2.6592

training:
  batch_size: 64
  learning_rate: 0.0001
  optimizer: "adamw"            # adamw | adam | sgd
  scheduler: "cosine"           # cosine | linear | constant
  loss_type: "reglip_regression"  # or "siglip_binary"
  max_epochs: 10
  warmup_steps: 500
  gradient_clip_norm: 1.0

data:
  dataset: "flickr30k"          # or "rsicd"
  data_root: "./data/flickr30k"
  use_regression_targets: true
  generate_similarities_online: true

logging:
  use_wandb: false
  use_tensorboard: true

checkpointing:
  checkpoint_dir: "./checkpoints/reglip"

device: "cuda"
mixed_precision: true
```

## Dependencies

Core: `torch>=2.0`, `transformers>=4.30`, `numpy`, `Pillow`, `pyyaml`, `tqdm`, `requests`
Training: `tensorboard`, `wandb`, `scikit-learn`
Optional: `sentence-transformers` (legacy), `qwen-omni-utils[decord]` + `flash-attn` (OmniEmbed)

Install: `pip install -r requirements.txt`

## Environment

- `.env` contains `OPEN_ROUTER_API_KEY` (not committed)
- Qwen API endpoint is hardcoded in configs as `http://212.41.29.82:6010`

## Conventions

- Configs are YAML, one per experiment variant
- Training scripts accept `--config` pointing to a YAML file
- Commit messages: short lowercase, `[action] [domain]` style (e.g. "fix rsicd", "eval rsicd")
- No CI/CD, no Docker — local research workflow
- Shell scripts (`train.sh`, `evaluate.sh`, `train_rsicd.sh`) are the main entry points
- Thesis documentation lives in `md/` and `chapters/`
