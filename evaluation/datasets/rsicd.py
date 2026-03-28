"""RSICD (Remote Sensing Image Captioning Dataset) for evaluation."""

import os
import json
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torchvision import transforms

from .base import BaseEvalDataset


# RSICD has 30 scene categories
RSICD_CLASS_NAMES = [
    "airport", "bare land", "baseball field", "beach", "bridge",
    "center", "church", "commercial", "dense residential", "desert",
    "farmland", "forest", "industrial", "meadow", "medium residential",
    "mountain", "park", "parking", "playground", "pond",
    "port", "railway station", "resort", "river", "school",
    "sparse residential", "square", "stadium", "storage tanks", "viaduct",
]


class RSICDClassificationDataset(BaseEvalDataset):
    """RSICD dataset for zero-shot scene classification.

    Expected directory structure:
        rsicd/
        ├── RSICD_images/
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── dataset_rsicd.json

    The JSON file contains image filenames and captions. Scene categories
    are inferred from the image filename prefix (e.g., "airport_001.jpg" → "airport").
    """

    name = "rsicd"
    task_type = "classification"

    CLASS_NAMES = RSICD_CLASS_NAMES

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        image_processor=None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor

        self.samples = self._load_samples()

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded RSICD {split} set with {len(self.samples)} samples "
              f"({len(set(s['label'] for s in self.samples))} classes)")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from dataset_rsicd.json."""
        json_path = os.path.join(self.data_root, "dataset_rsicd.json")
        images_dir = os.path.join(self.data_root, "RSICD_images")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"RSICD annotation file not found at {json_path}. "
                f"Expected structure:\n"
                f"  {self.data_root}/\n"
                f"  ├── RSICD_images/\n"
                f"  └── dataset_rsicd.json"
            )

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build class name to index mapping
        class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}

        samples = []
        skipped = 0

        for item in data["images"]:
            # Filter by split
            if item.get("split", "train") != self.split:
                continue

            filename = item["filename"]
            image_path = os.path.join(images_dir, filename)

            if not os.path.exists(image_path):
                skipped += 1
                continue

            # Extract scene category from filename
            # RSICD filenames follow pattern: "category_NNN.jpg"
            # e.g., "airport_001.jpg", "dense_residential_012.jpg"
            category = self._filename_to_category(filename)

            if category not in class_to_idx:
                skipped += 1
                continue

            samples.append({
                "image_path": image_path,
                "label": class_to_idx[category],
                "class_name": category,
                "filename": filename,
            })

        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing files or unknown categories)")

        return samples

    def _filename_to_category(self, filename: str) -> str:
        """Extract scene category from RSICD filename.

        Handles both simple ("airport_001.jpg") and compound
        ("dense_residential_012.jpg") category names.
        """
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Remove trailing digits and underscores (the image number)
        # e.g., "airport_001" → "airport", "dense_residential_012" → "dense_residential"
        parts = name.split("_")

        # The last part is always the number, remove it
        if parts[-1].isdigit():
            parts = parts[:-1]

        category = "_".join(parts)

        # Map underscores to spaces for matching with CLASS_NAMES
        category = category.replace("_", " ")

        # Handle special cases
        category_map = {
            "bareland": "bare land",
            "baseballfield": "baseball field",
            "denseresidential": "dense residential",
            "mediumresidential": "medium residential",
            "sparseresidential": "sparse residential",
            "railwaystation": "railway station",
            "storagetanks": "storage tanks",
        }

        # Try without spaces too
        no_space = category.replace(" ", "")
        if no_space in category_map:
            category = category_map[no_space]

        return category

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            pixel_values = transform(image)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(sample["label"]),
            "class_name": sample["class_name"],
        }

    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES


class RSICDRetrievalDataset(BaseEvalDataset):
    """RSICD dataset for image-text retrieval evaluation."""

    name = "rsicd_retrieval"
    task_type = "retrieval"

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        image_processor=None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor

        self.samples = self._load_samples()

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded RSICD retrieval {split} set with {len(self.samples)} samples")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load image-caption pairs from dataset_rsicd.json."""
        json_path = os.path.join(self.data_root, "dataset_rsicd.json")
        images_dir = os.path.join(self.data_root, "RSICD_images")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"RSICD annotation file not found at {json_path}"
            )

        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []

        for item in data["images"]:
            if item.get("split", "train") != self.split:
                continue

            filename = item["filename"]
            image_path = os.path.join(images_dir, filename)

            if not os.path.exists(image_path):
                continue

            # Use first caption for retrieval
            captions = [s["raw"] for s in item.get("sentences", [])]
            if not captions:
                continue

            samples.append({
                "image_path": image_path,
                "caption": captions[0],
                "all_captions": captions,
                "image_id": os.path.splitext(filename)[0],
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            pixel_values = image

        return {
            "pixel_values": pixel_values,
            "caption": sample["caption"],
            "image_id": sample["image_id"],
        }


class RSICDTrainingDataset(torch.utils.data.Dataset):
    """RSICD dataset for training (fine-tuning) with image-caption pairs.

    Returns raw image-caption pairs suitable for the training pipeline.
    Each image has 5 captions — one is randomly selected per epoch.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_processor=None,
        text_processor=None,
        max_samples: Optional[int] = None,
        random_caption: bool = True,
    ):
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.random_caption = random_caption

        self.samples = self._load_samples()

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded RSICD training {split} set with {len(self.samples)} image-caption groups")

    def _load_samples(self) -> List[Dict[str, Any]]:
        json_path = os.path.join(self.data_root, "dataset_rsicd.json")
        images_dir = os.path.join(self.data_root, "RSICD_images")

        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []

        for item in data["images"]:
            if item.get("split", "train") != self.split:
                continue

            filename = item["filename"]
            image_path = os.path.join(images_dir, filename)

            if not os.path.exists(image_path):
                continue

            captions = [s["raw"] for s in item.get("sentences", [])]
            if not captions:
                continue

            samples.append({
                "image_path": image_path,
                "captions": captions,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import random

        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        # Pick caption
        if self.random_caption:
            caption = random.choice(sample["captions"])
        else:
            caption = sample["captions"][0]

        # Process image
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            pixel_values = transform(image)

        return {
            "pixel_values": pixel_values,
            "caption": caption,
        }
