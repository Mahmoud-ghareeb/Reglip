"""PatternNet remote sensing dataset for zero-shot classification evaluation.

Uses the HuggingFace dataset: blanchon/PatternNet
38 classes, 256x256 RGB satellite images.
"""

from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torchvision import transforms

from .base import BaseEvalDataset


PATTERNNET_CLASS_NAMES = [
    "airplane", "baseball field", "basketball court", "beach", "bridge",
    "cemetery", "chaparral", "christmas tree farm", "closed road",
    "coastal mansion", "crosswalk", "dense residential", "ferry terminal",
    "football field", "forest", "freeway", "golf course", "harbor",
    "intersection", "mobile home park", "nursing home", "oil gas field",
    "oil well", "overpass", "parking lot", "parking space", "railway",
    "river", "runway", "runway marking", "shipping yard", "solar panel",
    "sparse residential", "storage tank", "swimming pool", "tennis court",
    "transformer station", "wastewater treatment plant",
]


class PatternNetDataset(BaseEvalDataset):
    """PatternNet dataset for zero-shot scene classification.

    Loads from HuggingFace: blanchon/PatternNet
    Supports selecting a subset of samples via max_samples.
    """

    name = "patternnet"
    task_type = "classification"
    CLASS_NAMES = PATTERNNET_CLASS_NAMES

    def __init__(
        self,
        data_root: str = None,
        split: str = "train",
        image_processor=None,
        max_samples: Optional[int] = None,
        download: bool = True,
    ):
        self.image_processor = image_processor

        from datasets import load_dataset

        dataset = load_dataset("blanchon/PatternNet", split=split)

        # Subsample if requested
        if max_samples and max_samples < len(dataset):
            # Stratified sampling: pick evenly across classes
            dataset = dataset.shuffle(seed=42).select(range(max_samples))

        self.dataset = dataset

        print(f"Loaded PatternNet {split} set with {len(self.dataset)} samples "
              f"({len(set(self.dataset['label']))} classes)")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        label = item["label"]

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
            "label": torch.tensor(label),
            "class_name": self.CLASS_NAMES[label],
        }

    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES
