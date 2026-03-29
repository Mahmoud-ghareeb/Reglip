#!/bin/bash
# Train and evaluate RegLIP vs SigLIP on RSICD (Remote Sensing domain)
#
# Usage:
#   ./train_rsicd.sh reglip          # Train RegLIP on RSICD
#   ./train_rsicd.sh siglip          # Train SigLIP on RSICD
#   ./train_rsicd.sh both            # Train both
#   ./train_rsicd.sh eval            # Evaluate both trained models
#
# Set RSICD_ROOT environment variable or pass --data_root to override default path.

set -e

MODE=${1:-both}
SEED=${2:-42}
DATA_ROOT=${RSICD_ROOT:-./data/rsicd}

echo "==========================================="
echo "RSICD Domain Experiment"
echo "Mode: $MODE | Seed: $SEED"
echo "Data root: $DATA_ROOT"
echo "==========================================="

train_reglip() {
    echo ""
    echo ">>> Training RegLIP on RSICD..."
    python training/train_rsicd.py \
        --config configs/reglip_rsicd_config.yaml \
        --data_root "$DATA_ROOT" \
        --seed "$SEED"
}

train_siglip() {
    echo ""
    echo ">>> Training SigLIP on RSICD..."
    python training/train_rsicd.py \
        --config configs/siglip_rsicd_config.yaml \
        --data_root "$DATA_ROOT" \
        --seed "$SEED"
}

evaluate() {
    echo ""
    echo ">>> Evaluating models on PatternNet zero-shot classification..."

    # RegLIP RSICD
    if [ -f checkpoints/reglip_rsicd/best_model.pth ]; then
        echo "  Evaluating RegLIP RSICD..."
        python scripts/run_evaluation.py \
            --checkpoint checkpoints/reglip_rsicd/best_model.pth \
            --model_type reglip \
            --model_name "RegLIP RSICD" \
            --dataset patternnet \
            --task zero_shot \
            --max_samples 1000 \
            --output results/reglip_patternnet
    fi

    # SigLIP RSICD
    if [ -f checkpoints/siglip_rsicd/best_model.pth ]; then
        echo "  Evaluating SigLIP RSICD..."
        python scripts/run_evaluation.py \
            --checkpoint checkpoints/siglip_rsicd/best_model.pth \
            --model_type reglip \
            --model_name "SigLIP RSICD" \
            --dataset patternnet \
            --task zero_shot \
            --max_samples 1000 \
            --output results/siglip_patternnet
    fi

    # Base pretrained (no fine-tuning)
    echo "  Evaluating Base pretrained on PatternNet..."
    python scripts/run_evaluation.py \
        --checkpoint none \
        --model_type reglip \
        --model_name "Base (pretrained)" \
        --dataset patternnet \
        --task zero_shot \
        --max_samples 1000 \
        --output results/base_patternnet

    echo ""
    echo ">>> Done! Results saved to results/"
}

case $MODE in
    reglip)
        train_reglip
        ;;
    siglip)
        train_siglip
        ;;
    both)
        train_reglip
        train_siglip
        ;;
    eval)
        evaluate
        ;;
    all)
        train_reglip
        train_siglip
        evaluate
        ;;
    *)
        echo "Usage: $0 {reglip|siglip|both|eval|all} [seed]"
        exit 1
        ;;
esac
