# RegLIP Project Structure

RegLIP is a regression-based vision-language model that enhances embeddings through continuous similarity learning instead of binary classification.

## 📁 Core Package Structure

### `reglip/` - Main Package
```
reglip/
├── __init__.py          # Package exports (RegLIPModel, RegLIPConfig, etc.)
├── model.py             # Core RegLIPModel implementation with RegLIPOutput
├── config.py            # Configuration classes (RegLIPConfig, RegLIPTextConfig, RegLIPVisionConfig)
├── text.py              # Text encoder implementation
├── vision.py            # Vision encoder implementation
├── utils.py             # Model utilities (SigLIP loading, frozen encoder setup)
└── embedding_utils.py   # Text embedding utilities (QwenEmbeddingClient, similarity)
```

**Key Classes:**
- `RegLIPModel`: Main model class with regression-based contrastive learning
- `RegLIPOutput`: Output dataclass with `text_embeds`, `image_embeds`, logits, loss, etc.
- `RegLIPConfig`: Model configuration with text/vision configs

### `training/` - Training Infrastructure
```
training/
├── __init__.py          # Training package exports
├── trainer.py           # BaseTrainer, SigLIPTrainer, RegLIPTrainer classes
├── train_reglip.py      # RegLIP training script
├── train_siglip.py      # SigLIP training script (for comparison)
└── utils.py             # Shared training utilities (optimizer/scheduler creation, logging)
```

**Consolidated Utilities:**
- `create_optimizer_and_scheduler()` - Unified optimizer/scheduler creation
- `setup_logging()` - Logging configuration
- `save_checkpoint()` / `load_checkpoint()` - Model checkpointing

### `data/` - Data Processing
```
data/
├── __init__.py          # Data package exports
├── dataset.py           # Dataset classes for vision-language data
├── preprocessing.py     # Data preprocessing utilities
└── flickr30k/          # Dataset-specific files
```

### `evaluation/` - Model Evaluation
```
evaluation/
├── __init__.py          # Evaluation exports
└── metrics.py           # Evaluation metrics and benchmarking
```

### `configs/` - Configuration Files
```
configs/
├── reglip_config.yaml   # RegLIP training configuration
└── siglip_config.yaml   # SigLIP training configuration
```

## 🚀 Entry Points & Scripts

### Main Scripts
- `example.py` - **Comprehensive examples** (basic usage, SigLIP loading, regression loss, embeddings)
- `train_comparison.py` - **Model comparison training** (train both SigLIP and RegLIP)
- `requirements.txt` - **Dependencies**

### Training Commands
```bash
# Train RegLIP model
python training/train_reglip.py --config configs/reglip_config.yaml

# Train SigLIP model (for comparison)
python training/train_siglip.py --config configs/siglip_config.yaml

# Train both models for comparison
python train_comparison.py --model both --data_root ./data/flickr30k
```

### Example Usage
```bash
# Run all examples
python example.py

# Basic model usage
from reglip import RegLIPModel, RegLIPConfig
model = RegLIPModel(RegLIPConfig())
outputs = model(input_ids=..., pixel_values=...)
text_embeds = outputs.text_embeds  # ✅ Available!
image_embeds = outputs.image_embeds  # ✅ Available!
```

## 🗂️ Supporting Directories

### Generated/Runtime Directories
```
checkpoints/             # Model checkpoints
logs/                   # Training logs
experiments/            # Experiment results
```

### Excluded from Indexing
```
transformers__/         # External transformers library (not indexed per request)
__pycache__/           # Python cache files
```

## 🔧 Key Features & Improvements

### ✅ Code Quality Improvements
1. **Removed Redundant Code**:
   - Consolidated duplicate `create_optimizer_and_scheduler()` functions
   - Merged example files (`example_embedding.py`, `test_siglip_loading.py` → `example.py`)
   - Removed backup files (`model.py.backup`)

2. **Enhanced Model Output**:
   - Added `RegLIPOutput` dataclass for structured outputs
   - Returns `text_embeds` and `image_embeds` properly
   - Follows HuggingFace transformers patterns

3. **Improved Training Infrastructure**:
   - Unified training utilities across SigLIP and RegLIP
   - Consolidated configuration management
   - Streamlined optimizer/scheduler creation

### 🎯 Usage Patterns

#### Model Creation & Forward Pass
```python
from reglip import RegLIPModel, RegLIPConfig

config = RegLIPConfig()
model = RegLIPModel(config)

outputs = model(input_ids=tokens, pixel_values=images)
# outputs.text_embeds, outputs.image_embeds, outputs.loss, etc.
```

#### Training
```python
from training.trainer import RegLIPTrainer
from training.utils import create_optimizer_and_scheduler

trainer = RegLIPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    logger=logger
)
trainer.train()
```

#### Configuration
```yaml
# configs/reglip_config.yaml
model:
  pretrained_model: "google/siglip-base-patch16-224"
  load_pretrained: true
  
training:
  optimizer: "adamw"
  learning_rate: 1e-4
  scheduler: "cosine"
```

## 📊 Architecture Overview

```
Text Input → RegLIPTextModel → text_embeds ↘
                                           RegLIP Loss (Regression)
Image Input → RegLIPVisionModel → image_embeds ↗

Frozen Text Encoder → Similarity Targets (for regression)
```

**Key Innovation**: Uses continuous similarity scores (0.0-1.0) instead of binary labels (0/1) for more nuanced vision-language learning.

## 🧹 Cleanup Summary

**Files Removed:**
- `reglip/model.py.backup` - Redundant backup
- `example_embedding.py` - Consolidated into main example
- `test_siglip_loading.py` - Functionality moved to examples
- Old log files - Cleaned up logs directory

**Code Consolidated:**
- Training utilities unified across scripts
- Example functionality merged into single file
- Duplicate optimizer/scheduler creation removed

**Structure Improved:**
- Clear separation of concerns
- Consistent import patterns
- Better configuration management
- Enhanced output structures

This cleaned and indexed structure provides a solid foundation for RegLIP development and usage. 