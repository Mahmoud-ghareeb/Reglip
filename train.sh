#!/bin/bash

SEED=${2:-42}
MODE=${1:-"siglip"}

case $MODE in
    "siglip")
        echo "Training SigLIP baseline (seed=$SEED)..."
        python training/train_siglip.py --config configs/siglip_config.yaml --seed $SEED
        ;;
    "reglip")
        echo "Training RegLIP (seed=$SEED)..."
        python training/train_reglip.py --config configs/reglip_config.yaml --seed $SEED
        ;;
    "both")
        echo "Training SigLIP then RegLIP (seed=$SEED)..."
        python training/train_siglip.py --config configs/siglip_config.yaml --seed $SEED
        python training/train_reglip.py --config configs/reglip_config.yaml --seed $SEED
        ;;
    *)
        echo "Usage: ./train.sh [MODE] [SEED]"
        echo ""
        echo "Modes:"
        echo "  siglip  - Train SigLIP baseline (default)"
        echo "  reglip  - Train RegLIP model"
        echo "  both    - Train both models sequentially"
        echo ""
        echo "SEED: Random seed for reproducibility (default: 42)"
        echo ""
        echo "Examples:"
        echo "  ./train.sh siglip"
        echo "  ./train.sh reglip 123"
        echo "  ./train.sh both 42"
        exit 1
        ;;
esac
