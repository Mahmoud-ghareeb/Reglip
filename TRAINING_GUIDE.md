# Training Guide: SigLIP vs RegLIP Comparison

This guide provides step-by-step instructions for training and comparing SigLIP and RegLIP models on the Flickr30K dataset.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For GPU support, make sure you have CUDA installed
# and install the appropriate PyTorch version
```

### 2. Prepare Flickr30K Dataset

Download the Flickr30K dataset and organize it as follows:

```
data/flickr30k/
├── images/
│   ├── 1000092795.jpg
│   ├── 1000268201.jpg
│   └── ...
└── captions/
    ├── dataset_flickr30k.json
    └── ...
```

**Alternative formats supported:**
- CSV files: `results_train.csv`, `results_val.csv`, `results_test.csv`
- JSON files: `flickr30k_train.json`, `flickr30k_val.json`, `flickr30k_test.json`

### 3. Run Training Comparison

```bash
# Train both models for comparison (recommended)
python train_comparison.py --model both --data_root ./data/flickr30k

# Or train individual models
python train_comparison.py --model siglip --data_root ./data/flickr30k
python train_comparison.py --model reglip --data_root ./data/flickr30k
```

## 📊 Training Options

### Basic Training Commands

```bash
# Debug mode with limited data (for testing)
python train_comparison.py --model both --debug

# Custom training parameters
python train_comparison.py \
    --model both \
    --data_root ./data/flickr30k \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4

# Train specific model with custom config
python training/train_reglip.py --config configs/reglip_config.yaml --debug
```

### Configuration Options

The training uses YAML configuration files in the `configs/` directory:

- `configs/siglip_config.yaml` - SigLIP training configuration
- `configs/reglip_config.yaml` - RegLIP training configuration

Key parameters you can modify:

```yaml
training:
  batch_size: 64          # Batch size for training
  learning_rate: 1e-4     # Learning rate
  max_epochs: 10          # Number of training epochs
  weight_decay: 0.01      # Weight decay for regularization

data:
  data_root: "./data/flickr30k"  # Path to dataset
  max_samples: null              # Limit samples (null = all data)
  num_workers: 8                 # DataLoader workers

logging:
  use_wandb: true                # Enable Weights & Biases logging
  project_name: "siglip_vs_reglip"
  log_every_n_steps: 100
```

## 🔬 Key Differences: SigLIP vs RegLIP

| Aspect | SigLIP | RegLIP |
|--------|--------|--------|
| **Loss Function** | Binary cross-entropy | Mean squared error (regression) |
| **Training Targets** | Binary (0/1) | Continuous (0.0-1.0) |
| **Similarity Encoding** | Rigid matching | Semantic gradients |
| **Architecture** | Text + Vision encoders | Text + Vision + Frozen text encoder |
| **Training Signal** | Limited | Rich and nuanced |

### RegLIP Innovation

RegLIP introduces **regression-based contrastive learning**:

1. **Continuous Similarity Targets**: Instead of binary 0/1 labels, uses continuous similarity scores
2. **Semantic Awareness**: Captures gradual relationships (e.g., "dog running" vs "animal playing" = 0.7 similarity)
3. **Automatic Label Generation**: Uses frozen sentence transformer to generate similarity targets
4. **Richer Training Signal**: More informative than binary classification

## 📈 Monitoring Training

### Weights & Biases (Recommended)

If W&B is enabled in config:

```bash
# View training in browser
wandb login  # First time only
# Training metrics will be automatically logged to W&B
```

### Local Logging

Training logs are saved to:
- `./logs/` - Training logs
- `./experiments/` - Experiment results and checkpoints

### Key Metrics to Monitor

**Training Metrics:**
- `train/loss` - Training loss
- `train/learning_rate` - Learning rate schedule
- `train/gradient_norm` - Gradient norm (for stability)

**RegLIP-specific Metrics:**
- `train/similarity_mean` - Mean similarity target
- `train/similarity_std` - Similarity target variance
- `train/regression_loss` - Regression loss component

**Validation Metrics:**
- `val/loss` - Validation loss
- `val/i2t_recall@1` - Image-to-text retrieval accuracy
- `val/t2i_recall@1` - Text-to-image retrieval accuracy

## 💾 Checkpoints and Results

### Directory Structure

```
experiments/siglip_vs_reglip_20241215_143022/
├── configs/
│   ├── siglip_config.yaml
│   └── reglip_config.yaml
├── checkpoints_siglip/
│   ├── best_model.pth
│   ├── checkpoint_epoch_1.pth
│   └── ...
├── checkpoints_reglip/
│   ├── best_model.pth
│   ├── checkpoint_epoch_1.pth
│   └── ...
└── experiment_summary.json
```

### Loading Trained Models

```python
import torch
from reglip import RegLIPModel, RegLIPConfig
from transformers import SiglipModel

# Load RegLIP model
checkpoint = torch.load("path/to/best_model.pth")
config = RegLIPConfig()  # or load from saved config
model = RegLIPModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Load SigLIP model
model = SiglipModel.from_pretrained("path/to/checkpoint/dir")
```

## 🎯 Expected Results

### Performance Improvements with RegLIP

Based on the research hypothesis, RegLIP should provide:

1. **Better Retrieval Performance**:
   - Higher Recall@1, Recall@5, Recall@10
   - Lower mean rank in retrieval tasks

2. **Improved Zero-shot Classification**:
   - Better understanding of semantic relationships
   - More nuanced similarity scoring

3. **Enhanced Embedding Quality**:
   - More semantically meaningful embedding spaces
   - Better fine-grained similarity understanding

### Sample Results Format

```
TRAINING RESULTS SUMMARY
============================================================

SIGLIP:
  Status: ✓ Success
  Training time: 2.34 hours
  Best val loss: 0.1234

REGLIP:
  Status: ✓ Success  
  Training time: 2.67 hours
  Best val loss: 0.1098

OVERALL:
  Successful models: 2/2
  Total training time: 5.01 hours

🎉 Both models trained successfully! You can now compare their performance.
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_comparison.py --batch_size 16 --model both

# Or use gradient accumulation in config
```

**2. Dataset Not Found**
```bash
# The system will create dummy data for testing
# Check console output for data loading status
```

**3. Import Errors**
```bash
# Install missing dependencies
pip install transformers sentence-transformers wandb

# For development
pip install -e .
```

**4. Training Slow**
```bash
# Reduce number of workers if on CPU
# Use mixed precision (enabled by default)
# Check GPU utilization with nvidia-smi
```

### Debug Mode

For quick testing:

```bash
# Run with limited data and shorter training
python train_comparison.py --debug --model both

# This sets:
# - max_samples: 200
# - max_epochs: 2  
# - batch_size: 8
# - disables W&B logging
```

## 📋 Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (e.g., RTX 3070, V100)
- **RAM**: 16GB system RAM
- **Storage**: 10GB for dataset + checkpoints

### Recommended Requirements
- **GPU**: 16GB+ VRAM (e.g., RTX 4080, A100)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ for full experiments

### CPU-Only Training
```bash
# Set device to CPU in config
device: "cpu"

# Note: Training will be significantly slower
```

## 🚀 Advanced Usage

### Custom Similarity Functions

You can modify the similarity generation in `data/preprocessing.py`:

```python
class SimilarityTargetGenerator:
    def generate_similarities(self, texts: List[str]) -> torch.Tensor:
        # Custom similarity logic here
        pass
```

### Custom Loss Functions

Modify the loss computation in `training/trainer.py`:

```python
class RegLIPTrainer(BaseTrainer):
    def compute_loss(self, batch):
        # Custom loss computation
        pass
```

### Distributed Training

For multi-GPU training, modify the config:

```yaml
distributed: true
device: "cuda"
```

## 📊 Evaluation and Comparison

After training, evaluate model performance:

```bash
# Compare trained models
python evaluation/compare_models.py \
    --siglip_checkpoint experiments/.../checkpoints_siglip/best_model.pth \
    --reglip_checkpoint experiments/.../checkpoints_reglip/best_model.pth \
    --data_root ./data/flickr30k

# Generate detailed analysis
python evaluation/analyze_results.py --exp_dir experiments/siglip_vs_reglip_20241215_143022/
```

## 🎓 Research Insights

### Hypothesis Testing

The RegLIP approach tests the hypothesis that:
> **Continuous similarity targets provide richer training signals than binary classification for vision-language learning**

### Key Research Questions

1. **Does regression improve retrieval performance?**
   - Compare Recall@k metrics between SigLIP and RegLIP

2. **Are embeddings more semantically meaningful?**
   - Analyze embedding quality and similarity distributions

3. **How does training efficiency compare?**
   - Compare convergence speed and final performance

### Publication-Ready Results

The training framework generates data suitable for:
- Academic papers on vision-language learning
- Comparison studies on contrastive learning approaches
- Analysis of semantic similarity in multimodal models

## 📞 Support

If you encounter issues:

1. **Check the logs** in `./logs/` directory
2. **Try debug mode** first: `--debug`
3. **Verify dataset format** matches expected structure
4. **Check GPU memory** usage with `nvidia-smi`

For research collaboration or technical questions, refer to the main README or create an issue in the repository. 