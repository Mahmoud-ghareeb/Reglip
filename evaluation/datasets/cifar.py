"""CIFAR datasets for zero-shot classification evaluation."""

import os
from typing import Dict, Any, Optional, List
import torch
from torchvision import datasets, transforms
from PIL import Image

from .base import BaseEvalDataset


class CIFAR10Dataset(BaseEvalDataset):
    """CIFAR-10 dataset for zero-shot classification."""
    
    name = "cifar10"
    task_type = "classification"
    
    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(
        self,
        data_root: str = "./data",
        split: str = "test",
        image_processor = None,
        download: bool = True,
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            data_root: Root directory for data
            split: 'train' or 'test'
            image_processor: Image preprocessing function
            download: Whether to download if not present
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load CIFAR-10
        train = (split == "train")
        self.dataset = datasets.CIFAR10(
            root=data_root,
            train=train,
            download=download,
        )
        
        print(f"Loaded CIFAR-10 {split} set with {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]
        
        # Convert to RGB if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        # Process image
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label),
            'class_name': self.CLASS_NAMES[label],
        }
    
    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES


class CIFAR100Dataset(BaseEvalDataset):
    """CIFAR-100 dataset for zero-shot classification."""
    
    name = "cifar100"
    task_type = "classification"
    
    # CIFAR-100 fine-grained class names
    CLASS_NAMES = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm"
    ]
    
    def __init__(
        self,
        data_root: str = "./data",
        split: str = "test",
        image_processor = None,
        download: bool = True,
    ):
        """
        Initialize CIFAR-100 dataset.
        
        Args:
            data_root: Root directory for data
            split: 'train' or 'test'
            image_processor: Image preprocessing function
            download: Whether to download if not present
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load CIFAR-100
        train = (split == "train")
        self.dataset = datasets.CIFAR100(
            root=data_root,
            train=train,
            download=download,
        )
        
        print(f"Loaded CIFAR-100 {split} set with {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]
        
        # Convert to RGB if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        # Process image
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(label),
            'class_name': self.CLASS_NAMES[label],
        }
    
    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES
