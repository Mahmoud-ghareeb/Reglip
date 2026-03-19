#!/bin/bash

# Evaluation script for RegLIP

# Default paths
REGLIP_CHECKPOINT="./checkpoints/reglip/best_model.pth"
SIGLIP_CHECKPOINT="./checkpoints/siglip/best_model.pth"
DATA_ROOT="${DATA_ROOT:-./data/flickr30k}"
OUTPUT_DIR="./results"

# Dataset paths (update these for your setup)
IMAGENET_ROOT="${IMAGENET_ROOT:-./data/imagenet-1k}"
IMAGENET_V2_ROOT="${IMAGENET_V2_ROOT:-./data/imagenetv2-matched-frequency-format-val}"
# OBJECTNET_ROOT="/home/mahmoud/RegLIP/data/objectnet"
# COCO_ROOT="${COCO_ROOT:-/path/to/coco}"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "RegLIP Evaluation Script"
echo "=============================================="

# Parse command line arguments
MODE=${1:-"reglip"}  # Default to reglip

case $MODE in
    "reglip")
        echo "Evaluating RegLIP model..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task all \
            --dataset all \
            --data_root $DATA_ROOT \
            --imagenet_root $IMAGENET_ROOT \
            --imagenet_v2_root $IMAGENET_V2_ROOT \
            --output $OUTPUT_DIR/reglip_eval \
            --format csv
        ;;
    
    "reglip_base")
        echo "Evaluating BASE RegLIP model (pretrained, no fine-tuning)..."
        python scripts/run_evaluation.py \
            --checkpoint "" \
            --model_type reglip \
            --task all \
            --dataset all \
            --data_root $DATA_ROOT \
            --imagenet_root $IMAGENET_ROOT \
            --imagenet_v2_root $IMAGENET_V2_ROOT \
            --output $OUTPUT_DIR/reglip_base_eval \
            --format csv
        ;;
    
    "siglip")
        echo "Evaluating SigLIP model..."
        python scripts/run_evaluation.py \
            --checkpoint $SIGLIP_CHECKPOINT \
            --model_type siglip \
            --task all \
            --dataset all \
            --data_root $DATA_ROOT \
            --imagenet_root $IMAGENET_ROOT \
            --imagenet_v2_root $IMAGENET_V2_ROOT \
            --output $OUTPUT_DIR/siglip_eval \
            --format csv
        ;;
    
    "siglip_base")
        echo "Evaluating BASE SigLIP model (pretrained, no fine-tuning)..."
        python scripts/run_evaluation.py \
            --checkpoint "" \
            --model_type siglip \
            --task all \
            --dataset all \
            --data_root $DATA_ROOT \
            --imagenet_root $IMAGENET_ROOT \
            --imagenet_v2_root $IMAGENET_V2_ROOT \
            --output $OUTPUT_DIR/siglip_base_eval \
            --format csv
        ;;
    
    "compare")
        echo "Comparing RegLIP vs SigLIP..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT $SIGLIP_CHECKPOINT \
            --model_names RegLIP SigLIP \
            --compare \
            --task all \
            --dataset all \
            --data_root $DATA_ROOT \
            --imagenet_root $IMAGENET_ROOT \
            --imagenet_v2_root $IMAGENET_V2_ROOT \
            --output $OUTPUT_DIR/comparison \
            --format json csv latex
        ;;
    
    "retrieval")
        echo "Running retrieval evaluation only..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task retrieval \
            --dataset flickr30k \
            --data_root $DATA_ROOT \
            --output $OUTPUT_DIR/retrieval_eval
        ;;
    
    "zero_shot")
        echo "Running zero-shot evaluation only..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task zero_shot \
            --dataset cifar10 cifar100 \
            --output $OUTPUT_DIR/zero_shot_eval
        ;;
    
    "imagenet")
        echo "Running ImageNet-1k evaluation (like SigLIP paper)..."
        python scripts/run_evaluation.py \
            --checkpoint "" \
            --model_type siglip \
            --task zero_shot \
            --dataset imagenet \
            --imagenet_root $IMAGENET_ROOT \
            --output $OUTPUT_DIR/imagenet_eval \
            --format csv
        ;;
    
    "imagenet_all")
        echo "Running all ImageNet variants (Validation, V2, ReaL, ObjectNet)..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task zero_shot \
            --dataset imagenet imagenet_v2 imagenet_real objectnet \
            --data_root $IMAGENET_ROOT \
            --output $OUTPUT_DIR/imagenet_all_eval \
            --format csv
        ;;
    
    "coco")
        echo "Running MS-COCO retrieval evaluation..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task retrieval \
            --dataset coco \
            --data_root $COCO_ROOT \
            --output $OUTPUT_DIR/coco_eval \
            --format json csv latex
        ;;
    
    "siglip_paper")
        echo "Running SigLIP paper benchmarks (ImageNet-1k + COCO)..."
        echo ""
        echo "ImageNet-1k variants..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task zero_shot \
            --dataset imagenet imagenet_v2 objectnet \
            --data_root $IMAGENET_ROOT \
            --output $OUTPUT_DIR/siglip_paper_imagenet \
            --format csv
        echo ""
        echo "COCO Retrieval..."
        python scripts/run_evaluation.py \
            --checkpoint $REGLIP_CHECKPOINT \
            --task retrieval \
            --dataset coco \
            --data_root $COCO_ROOT \
            --output $OUTPUT_DIR/siglip_paper_coco \
            --format csv
        ;;
    
    *)
        echo "Usage: ./evaluate.sh [MODE]"
        echo ""
        echo "Basic Modes:"
        echo "  reglip        - Evaluate fine-tuned RegLIP model on all tasks"
        echo "  reglip_base   - Evaluate BASE RegLIP model (pretrained)"
        echo "  siglip        - Evaluate fine-tuned SigLIP model on all tasks"
        echo "  siglip_base   - Evaluate BASE SigLIP model (pretrained)"
        echo "  compare       - Compare RegLIP vs SigLIP"
        echo ""
        echo "Task-specific:"
        echo "  retrieval     - Run retrieval evaluation only (Flickr30K)"
        echo "  zero_shot     - Run zero-shot classification only (CIFAR)"
        echo ""
        echo "SigLIP Paper Benchmarks:"
        echo "  imagenet      - ImageNet-1k validation set"
        echo "  imagenet_all  - All ImageNet variants (Val, V2, ReaL, ObjectNet)"
        echo "  coco          - MS-COCO retrieval (I→T, T→I R@1)"
        echo "  siglip_paper  - Full SigLIP paper benchmarks (ImageNet + COCO)"
        echo ""
        echo "Environment variables for dataset paths:"
        echo "  IMAGENET_ROOT    - Path to ImageNet (default: /path/to/imagenet)"
        echo "  IMAGENET_V2_ROOT - Path to ImageNet-V2 (default: /path/to/imagenetv2-matched-frequency)"
        echo "  OBJECTNET_ROOT   - Path to ObjectNet (default: /path/to/objectnet)"
        echo "  COCO_ROOT        - Path to MS-COCO (default: /path/to/coco)"
        exit 1
        ;;
esac

echo ""
echo "Evaluation complete! Results saved to $OUTPUT_DIR"
