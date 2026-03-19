#!/usr/bin/env python3
"""
Download ImageNet validation set with labels.

Uses the approach from PyTorch_ImageNet_Validation_Example.ipynb:
- Clones Small-ImageNet-Validation-Dataset-1000-Classes (5 images per class, 5000 total)
- Structure: ILSVRC2012_img_val_subset/{class_index}/image.JPEG
- imagenet.json for 1000 class names
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes"
OUTPUT_DIR = Path("./data/imagenet-1k")
SUBSET_DIR = "ILSVRC2012_img_val_subset"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clone_dir = OUTPUT_DIR / "Small-ImageNet-Validation-Dataset-1000-Classes"

    # Step 1: Clone the repository
    if clone_dir.exists():
        print(f"Repository already exists at {clone_dir}")
    else:
        print("Cloning Small-ImageNet-Validation-Dataset-1000-Classes...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
            check=True,
        )

    # Step 2: Class labels - use imagenet-simple-labels (canonical for SigLIP/CLIP)
    # Notebook uses imagenet.json; we prefer simple-labels for correct zero-shot mapping
    import urllib.request
    simple_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    dst_json = OUTPUT_DIR / "imagenet_classes.json"
    try:
        urllib.request.urlretrieve(simple_url, str(dst_json))
        with open(dst_json) as f:
            class_names = json.load(f)
        print(f"Saved {len(class_names)} class names (imagenet-simple-labels) to {dst_json}")
    except Exception as e:
        print(f"Downloading simple-labels failed: {e}")
        # Fallback: use notebook's imagenet.json
        src_json = clone_dir / "imagenet.json"
        if src_json.exists():
            with open(src_json) as f:
                labels = json.load(f)
            class_names = [label.split(",")[0].strip() for label in labels]
            with open(dst_json, "w") as f:
                json.dump(class_names, f, indent=2)
            print(f"Saved {len(class_names)} class names from imagenet.json to {dst_json}")
        else:
            print(f"Warning: {src_json} not found")

    # Step 3: Move/copy ILSVRC2012_img_val_subset to output root
    src_subset = clone_dir / SUBSET_DIR
    dst_subset = OUTPUT_DIR / SUBSET_DIR

    if src_subset.exists():
        if dst_subset.exists():
            print(f"Validation subset already at {dst_subset}")
        else:
            shutil.copytree(src_subset, dst_subset)
            print(f"Copied validation subset to {dst_subset}")
    else:
        print(f"Warning: {src_subset} not found")

    # Step 4: Count samples
    total = 0
    if dst_subset.exists():
        for d in dst_subset.iterdir():
            if d.is_dir() and d.name.isdigit():
                total += len(list(d.glob("*.JPEG")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))

    print(f"\nDone! {total} images in {OUTPUT_DIR.resolve()}")
    print("Structure: data/imagenet-1k/ILSVRC2012_img_val_subset/{0..999}/image.JPEG")


if __name__ == "__main__":
    main()
