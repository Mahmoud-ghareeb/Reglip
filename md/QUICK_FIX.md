# Quick Fix Guide

This document provides solutions for common issues encountered when setting up and running RegLIP.

## 🔧 Dependency Issues

### Dataset Validation Issues

**Error:**
```
Quick dataset check failed!
Dataset validation failed!
```

**Solution:**
```bash
# 1. Validate your dataset
python validate_dataset.py

# 2. Check expected structure
# Your data should be organized like this:
# /home/mahmoud/RegLIP/data/flickr30k/
# ├── images/
# │   ├── 1000092795.jpg
# │   ├── 1000268201.jpg
# │   └── ...
# └── captions.txt

# 3. Ensure captions.txt format:
# Each line should be: filename,caption
# Example:
# image1.jpg,A person walking in the park
# image2.jpg,A dog playing with a ball

# 4. Quick validation options
python validate_dataset.py --quick-only        # Fast check
python validate_dataset.py --quiet            # Less verbose
python validate_dataset.py --data-root /path  # Custom path
```

**For missing dataset:**
```bash
# Continue training with dummy data (for testing)
python training/train_reglip.py --debug
python train_comparison.py --model both --debug
```

### SentencePiece Missing Error

**Error:**
```
SiglipTokenizer requires the SentencePiece library but it was not found in your environment.
```

**Solution:**
```bash
# Install sentencepiece
pip install sentencepiece>=0.1.99

# Or install all updated requirements
pip install -r requirements.txt
```

### Wandb 403 Forbidden Error

**Error:**
```
Error uploading run: returned error 403: {"data":null,"errors":[{"message":"403 Forbidden"}]}
```

**Solution:**
```bash
# Option 1: Disable wandb (recommended for beginners)
# Edit configs/reglip_config.yaml or configs/siglip_config.yaml
# Set: use_wandb: false

# Option 2: Setup wandb authentication
wandb login
# Then follow the prompts to enter your API key

# Option 3: Use offline mode
# The training will automatically fallback to offline mode if online fails
```

**Quick Fix:**
```bash
# Run training with wandb disabled
python training/train_reglip.py --debug  # Already disabled in default configs
```

### Complete Environment Setup

**For a fresh installation:**

```bash
# 1. Clone the repository
git clone <repository-url>
cd RegLIP

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python -c "
import torch
from transformers import SiglipProcessor
from reglip import RegLIPModel, RegLIPConfig
print('✅ All dependencies installed correctly!')
"
```

## 🚀 Quick Start After Fix

### Test the Fixed Setup

```bash
# Run examples to verify everything works
python example.py

# Test training scripts (dry run)
python training/train_siglip.py --debug
python training/train_reglip.py --debug
```

### Training Commands

```bash
# Train RegLIP model
python training/train_reglip.py --config configs/reglip_config.yaml --debug

# Train SigLIP model (for comparison)
python training/train_siglip.py --config configs/siglip_config.yaml --debug

# Train both models for comparison
python train_comparison.py --model both --debug
```

## 📊 Verification Checklist

Run this checklist to ensure everything is working:

```bash
# ✅ 1. Test imports
python -c "
from reglip import RegLIPModel, RegLIPConfig, RegLIPOutput
from training.utils import create_optimizer_and_scheduler
from data.preprocessing import create_siglip_processors
print('✅ All imports successful')
"

# ✅ 2. Test model creation
python -c "
from reglip import RegLIPModel, RegLIPConfig
model = RegLIPModel(RegLIPConfig())
print('✅ Model creation successful')
"

# ✅ 3. Test preprocessing
python -c "
from data.preprocessing import create_siglip_processors
img_proc, text_proc = create_siglip_processors()
print('✅ Preprocessing functions working')
"

# ✅ 4. Test forward pass
python -c "
import torch
from reglip import RegLIPModel, RegLIPConfig
model = RegLIPModel(RegLIPConfig())
outputs = model(
    input_ids=torch.randint(0, 1000, (2, 16)),
    pixel_values=torch.randn(2, 3, 224, 224),
    attention_mask=torch.ones(2, 16)
)
print(f'✅ Forward pass successful: {type(outputs)}')
print(f'✅ Embeddings available: text_embeds={hasattr(outputs, \"text_embeds\")}, image_embeds={hasattr(outputs, \"image_embeds\")}')
"
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

1. **CUDA out of memory**
   ```bash
   # Reduce batch size in configs
   # Edit configs/reglip_config.yaml or configs/siglip_config.yaml
   # Change batch_size from 32 to 16 or 8
   ```

2. **Transformers version conflicts**
   ```bash
   pip install transformers>=4.30.0 --upgrade
   ```

3. **Missing data directory**
   ```bash
   # Create data directory structure
   mkdir -p data/flickr30k/images
   mkdir -p data/flickr30k/captions
   ```

4. **Import errors**
   ```bash
   # Ensure you're in the project root directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

## 🎯 Next Steps

After resolving the dependency issues:

1. **Run examples**: `python example.py`
2. **Test training**: `python training/train_reglip.py --debug`
3. **Prepare your data**: Set up Flickr30K or your custom dataset
4. **Start training**: Remove `--debug` flag for full training

## 📝 Updated Features

The cleaned codebase now includes:

- ✅ **Fixed Dependencies**: All required packages in requirements.txt
- ✅ **Consolidated Training Utilities**: No more duplicate code
- ✅ **Enhanced Model Output**: Proper `RegLIPOutput` with embeddings
- ✅ **Unified Examples**: All examples in single `example.py`
- ✅ **Better Configuration**: YAML-based configs with validation

For detailed project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md). 