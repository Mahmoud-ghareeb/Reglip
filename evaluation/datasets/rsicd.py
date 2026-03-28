"""RSICD (Remote Sensing Image Captioning Dataset) for evaluation and training.

Expected directory structure:
    data/rsicd/
    ├── train.csv
    ├── test.csv
    └── valid.csv

Each CSV has columns: image, captions, filename (and possibly others).
- image: byte array string, parsed via ast.literal_eval to get {'bytes': ...}
- captions: newline-separated caption strings wrapped in brackets
"""

import os
import io
import ast
import random
import pandas as pd
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

# Map to split CSV filenames
_SPLIT_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "val": "valid.csv",
    "valid": "valid.csv",
}


def _bytes_to_image(byte_array_str: str) -> Image.Image:
    """Convert a byte array string from the CSV into a PIL Image."""
    data = ast.literal_eval(byte_array_str)
    return Image.open(io.BytesIO(data["bytes"])).convert("RGB")


def _parse_captions(captions_str: str) -> List[str]:
    """Parse the captions column into a list of strings."""
    cleaned = captions_str.replace("[", "").replace("]", "")
    captions = [c.strip().strip("'\"") for c in cleaned.split("\n") if c.strip()]
    return [c for c in captions if c]


def _load_csv(data_root: str, split: str) -> pd.DataFrame:
    """Load the CSV file for the given split."""
    filename = _SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Unknown split '{split}'. Expected: {list(_SPLIT_FILES.keys())}")

    csv_path = os.path.join(data_root, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"RSICD CSV not found at {csv_path}. "
            f"Expected structure:\n"
            f"  {data_root}/\n"
            f"  ├── train.csv\n"
            f"  ├── test.csv\n"
            f"  └── valid.csv"
        )
    return pd.read_csv(csv_path)


def _filename_to_category(filename: str) -> str:
    """Extract scene category from RSICD filename.

    Handles both simple ('airport_001.jpg') and compound
    ('denseresidential_012.jpg') category names.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    # Remove trailing numeric part
    if parts[-1].isdigit():
        parts = parts[:-1]

    category = "_".join(parts).lower()

    # Map known filename prefixes to CLASS_NAMES
    category_map = {
        "airport": "airport",
        "bareland": "bare land",
        "baseballfield": "baseball field",
        "beach": "beach",
        "bridge": "bridge",
        "center": "center",
        "church": "church",
        "commercial": "commercial",
        "denseresidential": "dense residential",
        "desert": "desert",
        "farmland": "farmland",
        "forest": "forest",
        "industrial": "industrial",
        "meadow": "meadow",
        "mediumresidential": "medium residential",
        "mountain": "mountain",
        "park": "park",
        "parking": "parking",
        "playground": "playground",
        "pond": "pond",
        "port": "port",
        "railwaystation": "railway station",
        "resort": "resort",
        "river": "river",
        "school": "school",
        "sparseresidential": "sparse residential",
        "square": "square",
        "stadium": "stadium",
        "storagetanks": "storage tanks",
        "viaduct": "viaduct",
    }

    # Try with and without underscores
    key = category.replace("_", "")
    if key in category_map:
        return category_map[key]

    # Fallback: replace underscores with spaces
    return category.replace("_", " ")


class RSICDClassificationDataset(BaseEvalDataset):
    """RSICD dataset for zero-shot scene classification.

    Loads images from CSV byte arrays. Scene categories are inferred
    from the filename column.
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

        self.df = _load_csv(data_root, split)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}

        # Build valid sample indices and labels
        self.indices = []
        self.labels = []
        self.class_names_per_sample = []

        skipped = 0
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            filename = str(row.get("filename", ""))
            if not filename:
                skipped += 1
                continue

            category = _filename_to_category(filename)
            if category not in self.class_to_idx:
                skipped += 1
                continue

            self.indices.append(i)
            self.labels.append(self.class_to_idx[category])
            self.class_names_per_sample.append(category)

        if max_samples:
            self.indices = self.indices[:max_samples]
            self.labels = self.labels[:max_samples]
            self.class_names_per_sample = self.class_names_per_sample[:max_samples]

        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing filename or unknown category)")

        n_classes = len(set(self.labels))
        print(f"Loaded RSICD {split} set with {len(self.indices)} samples ({n_classes} classes)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]

        image = _bytes_to_image(row["image"])

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
            "label": torch.tensor(self.labels[idx]),
            "class_name": self.class_names_per_sample[idx],
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

        self.df = _load_csv(data_root, split)

        if max_samples:
            self.df = self.df.head(max_samples)

        print(f"Loaded RSICD retrieval {split} set with {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image = _bytes_to_image(row["image"])

        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            pixel_values = image

        captions = _parse_captions(row["captions"])
        caption = captions[0] if captions else ""

        filename = str(row.get("filename", f"sample_{idx}"))
        image_id = os.path.splitext(filename)[0]

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_id": image_id,
        }


class RSICDTrainingDataset(torch.utils.data.Dataset):
    """RSICD dataset for training (fine-tuning) with image-caption pairs.

    Each image has multiple captions — one is randomly selected per access.
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

        self.df = _load_csv(data_root, split)

        if max_samples:
            self.df = self.df.head(max_samples)

        print(f"Loaded RSICD training {split} set with {len(self.df)} image-caption groups")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image = _bytes_to_image(row["image"])

        # Parse and pick caption
        captions = _parse_captions(row["captions"])
        if not captions:
            caption = ""
        elif self.random_caption:
            caption = random.choice(captions)
        else:
            caption = captions[0]

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
