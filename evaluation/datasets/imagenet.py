"""ImageNet datasets for zero-shot classification evaluation."""

import os
from typing import Dict, Any, Optional, List
import torch
from torchvision import transforms
from PIL import Image
import json

from .base import BaseEvalDataset

# ImageNet class names (1000 classes)
# This is a subset - full list would be loaded from file
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def _normalize_class_name(s: str) -> str:
    """Extract primary class name from 'tench, Tinca tinca' format."""
    return s.split(",")[0].strip()


def _try_load_json(path: str) -> Optional[List[str]]:
    """Try to load a JSON class names file. Returns None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 1000:
            # Handle "tench, Tinca tinca" format from PyTorch notebook
            return [_normalize_class_name(x) for x in data]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def load_imagenet_classes(data_root: str) -> List[str]:
    """Load ImageNet class names for zero-shot prompts.
    
    SigLIP/CLIP expect imagenet-simple-labels ordering. Prefer that for compatibility.
    Small-ImageNet (notebook) uses imagenet.json with same ordering; we normalize
    'tench, Tinca tinca' -> 'tench'.
    """
    # 1. Prefer imagenet-simple-labels (canonical for SigLIP/CLIP zero-shot)
    try:
        import urllib.request
        simple_path = os.path.join(data_root, "imagenet_simple_labels.json")
        if not os.path.exists(simple_path):
            urllib.request.urlretrieve(IMAGENET_CLASSES_URL, simple_path)
        result = _try_load_json(simple_path)
        if result is not None:
            return result
    except Exception:
        pass

    # 2. Local files: imagenet_classes.json, imagenet.json (notebook format)
    search_paths = [
        os.path.join(data_root, "imagenet_classes.json"),
        os.path.join(data_root, "imagenet.json"),
        os.path.join(data_root, "val", "imagenet_classes.json"),
        os.path.join(os.path.dirname(data_root), "imagenet_classes.json"),
    ]
    for path in search_paths:
        result = _try_load_json(path)
        if result is not None:
            return result

    # 3. Try to download to data_root
    classes_file = os.path.join(data_root, "imagenet_classes.json")
    try:
        import urllib.request
        os.makedirs(data_root, exist_ok=True)
        print(f"Downloading ImageNet class names to {classes_file}...")
        urllib.request.urlretrieve(IMAGENET_CLASSES_URL, classes_file)
        result = _try_load_json(classes_file)
        if result is not None:
            return result
    except Exception as e:
        print(f"Could not download class names: {e}")

    return [f"class_{i}" for i in range(1000)]


class ImageNetDataset(BaseEvalDataset):
    """ImageNet-1k validation dataset for zero-shot classification."""
    
    name = "imagenet"
    task_type = "classification"
    num_classes = 1000
    
    def __init__(
        self,
        data_root: str,
        split: str = "val",
        image_processor = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            data_root: Path to ImageNet directory (should contain 'val' folder)
            split: Dataset split ('val')
            image_processor: Image preprocessing function
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load class names
        self.class_names = load_imagenet_classes(data_root)
        
        # Load samples
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded ImageNet {split} set with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from ImageNet directory structure."""
        samples = []

        # PyTorch notebook structure: data_root/ILSVRC2012_img_val_subset/0/, 1/, ... 999/
        subset_dir = os.path.join(self.data_root, "ILSVRC2012_img_val_subset")
        if os.path.exists(subset_dir):
            print(f"Detected ILSVRC2012_img_val_subset format in {subset_dir}")
            for class_idx in range(1000):
                class_dir = os.path.join(subset_dir, str(class_idx))
                if not os.path.exists(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        samples.append({
                            'image_path': os.path.join(class_dir, img_name),
                            'label': class_idx,
                            'synset': str(class_idx),
                        })
            return samples

        # Try standard ImageNet structure: val/n01440764/ILSVRC2012_val_00000293.JPEG
        val_dir = os.path.join(self.data_root, self.split)

        # Check if we have numbered class folders (00000, 00001, ...) or (0, 1, ...) in data_root
        if os.path.exists(self.data_root):
            dirs = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
            # 5-digit format (00000-00999) or short format (0-999)
            digit_dirs = [d for d in dirs if d.isdigit()]
            if digit_dirs:
                # Numbered folder format: data_root/00000/image.jpg
                print(f"Detected numbered class folder format in {self.data_root}")
                
                # Check for custom folder-to-class mapping file
                mapping_file = os.path.join(self.data_root, "folder_to_class_mapping.json")
                folder_to_class = None
                if os.path.exists(mapping_file):
                    print(f"Loading folder-to-class mapping from {mapping_file}")
                    with open(mapping_file, 'r') as f:
                        folder_to_class = json.load(f)
                    # Convert string keys to int if needed
                    if folder_to_class and isinstance(list(folder_to_class.keys())[0], str):
                        folder_to_class = {int(k): v for k, v in folder_to_class.items()}
                
                for class_idx_str in sorted(digit_dirs, key=int):
                    folder_num = int(class_idx_str)
                    
                    # Use mapping if available, otherwise assume folder number = class index
                    if folder_to_class is not None:
                        class_idx = folder_to_class.get(folder_num, folder_num)
                    else:
                        class_idx = folder_num
                    
                    class_dir = os.path.join(self.data_root, class_idx_str)
                    
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            samples.append({
                                'image_path': os.path.join(class_dir, img_name),
                                'label': class_idx,
                                'synset': class_idx_str,
                            })
                return samples
        
        # Standard ImageNet structure: val/n01440764/ILSVRC2012_val_00000293.JPEG
        if not os.path.exists(val_dir):
            print(f"Warning: ImageNet {self.split} directory not found at {val_dir}")
            return samples
        
        # Load synset to class index mapping
        synset_file = os.path.join(self.data_root, "synset_to_idx.json")
        if os.path.exists(synset_file):
            with open(synset_file, 'r') as f:
                synset_to_idx = json.load(f)
        else:
            # Build mapping from directory names
            synset_to_idx = {}
            synsets = sorted(os.listdir(val_dir))
            for idx, synset in enumerate(synsets):
                if os.path.isdir(os.path.join(val_dir, synset)):
                    synset_to_idx[synset] = idx
        
        # Load all images
        for synset in sorted(os.listdir(val_dir)):
            synset_dir = os.path.join(val_dir, synset)
            if not os.path.isdir(synset_dir):
                continue
            
            label = synset_to_idx.get(synset, 0)
            
            for img_name in os.listdir(synset_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append({
                        'image_path': os.path.join(synset_dir, img_name),
                        'label': label,
                        'synset': synset,
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Process image
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(sample['label']),
            'class_name': self.class_names[sample['label']] if sample['label'] < len(self.class_names) else f"class_{sample['label']}",
        }
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class ImageNetV2Dataset(BaseEvalDataset):
    """ImageNet-V2 dataset for zero-shot classification (distribution shift test)."""
    
    name = "imagenet_v2"
    task_type = "classification"
    num_classes = 1000
    
    def __init__(
        self,
        data_root: str,
        split: str = "matched-frequency",  # matched-frequency, threshold0.7, topimages
        image_processor = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize ImageNet-V2 dataset.
        
        Download from: https://github.com/modestyachts/ImageNetV2
        
        Args:
            data_root: Path to ImageNet-V2 directory
            split: Dataset variant
            image_processor: Image preprocessing function
            max_samples: Maximum number of samples
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load class names (same as ImageNet) — check data_root, val/ subdir, then parent
        self.class_names = load_imagenet_classes(data_root)
        if self.class_names[0] == "class_0":
            val_dir = os.path.join(data_root, "val")
            if os.path.exists(os.path.join(val_dir, "imagenet_classes.json")):
                self.class_names = load_imagenet_classes(val_dir)
        if self.class_names[0] == "class_0":
            parent_dir = os.path.dirname(data_root)
            self.class_names = load_imagenet_classes(parent_dir)
        
        # Load samples
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded ImageNet-V2 ({split}) with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from ImageNet-V2 directory structure."""
        samples = []

        # ImageNet-V2 structure: 0/, 1/, ..., 999/ with images inside
        # Some downloads nest them under a val/ subdirectory
        search_root = self.data_root
        if not os.path.exists(search_root):
            print(f"Warning: ImageNet-V2 directory not found at {search_root}")
            return samples

        # Check if class folders are directly here or inside val/
        if os.path.isdir(os.path.join(search_root, "val")):
            candidate = os.path.join(search_root, "val")
            if os.path.isdir(os.path.join(candidate, "0")):
                search_root = candidate

        for class_idx in range(1000):
            class_dir = os.path.join(search_root, str(class_idx))
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append({
                        'image_path': os.path.join(class_dir, img_name),
                        'label': class_idx,
                    })

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(sample['label']),
            'class_name': self.class_names[sample['label']] if sample['label'] < len(self.class_names) else f"class_{sample['label']}",
        }
    
    def get_class_names(self) -> List[str]:
        return self.class_names


class ObjectNetDataset(BaseEvalDataset):
    """ObjectNet dataset for zero-shot classification (robustness test)."""
    
    name = "objectnet"
    task_type = "classification"
    
    # ObjectNet to ImageNet class mapping (subset of 113 overlapping classes)
    # This is a simplified version - full mapping should be loaded from file
    
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        image_processor = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize ObjectNet dataset.
        
        Download from: https://objectnet.dev/
        
        Args:
            data_root: Path to ObjectNet directory
            split: Dataset split
            image_processor: Image preprocessing function
            max_samples: Maximum number of samples
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load mapping and class names
        self._load_mappings()
        
        # Load samples
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded ObjectNet with {len(self.samples)} samples ({len(self.objectnet_to_imagenet)} classes)")
    
    def _load_mappings(self):
        """Load ObjectNet to ImageNet class mappings."""
        mapping_file = os.path.join(self.data_root, "objectnet_to_imagenet.json")
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.objectnet_to_imagenet = json.load(f)
        else:
            # Default mapping for common classes (simplified)
            self.objectnet_to_imagenet = {}
        
        # Load ImageNet class names
        parent_dir = os.path.dirname(self.data_root)
        self.imagenet_classes = load_imagenet_classes(parent_dir)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from ObjectNet directory."""
        samples = []
        
        images_dir = os.path.join(self.data_root, "images")
        if not os.path.exists(images_dir):
            images_dir = self.data_root
        
        if not os.path.exists(images_dir):
            print(f"Warning: ObjectNet directory not found at {images_dir}")
            return samples
        
        for class_name in os.listdir(images_dir):
            class_dir = os.path.join(images_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Skip if no ImageNet mapping
            if class_name not in self.objectnet_to_imagenet:
                continue
            
            imagenet_idx = self.objectnet_to_imagenet[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append({
                        'image_path': os.path.join(class_dir, img_name),
                        'label': imagenet_idx,
                        'objectnet_class': class_name,
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.image_processor:
            pixel_values = self.image_processor(image)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(sample['label']),
            'class_name': self.imagenet_classes[sample['label']] if sample['label'] < len(self.imagenet_classes) else sample['objectnet_class'],
        }
    
    def get_class_names(self) -> List[str]:
        return self.imagenet_classes


class ImageNetReaLDataset(ImageNetDataset):
    """
    ImageNet with ReaL labels for more accurate evaluation.
    
    ReaL (Re-Assessed Labels) provides multiple correct labels per image,
    accounting for ambiguity and labeling errors in original ImageNet.
    """
    
    name = "imagenet_real"
    
    def __init__(
        self,
        data_root: str,
        split: str = "val",
        image_processor = None,
        max_samples: Optional[int] = None,
    ):
        # First initialize parent
        super().__init__(data_root, split, image_processor, max_samples)
        
        # Load ReaL labels
        self.real_labels = self._load_real_labels()
        
        if self.real_labels:
            print(f"Loaded ReaL labels for {len(self.real_labels)} images")
    
    def _load_real_labels(self) -> Dict[str, List[int]]:
        """Load ReaL labels from file."""
        real_file = os.path.join(self.data_root, "real_labels.json")
        
        if os.path.exists(real_file):
            with open(real_file, 'r') as f:
                return json.load(f)
        
        # Try alternative location
        real_file = os.path.join(os.path.dirname(self.data_root), "real_labels.json")
        if os.path.exists(real_file):
            with open(real_file, 'r') as f:
                return json.load(f)
        
        print("Warning: ReaL labels not found. Using standard ImageNet labels.")
        return {}
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = super().__getitem__(idx)
        
        # Add ReaL labels if available
        sample = self.samples[idx]
        img_name = os.path.basename(sample['image_path'])
        
        if img_name in self.real_labels:
            result['real_labels'] = self.real_labels[img_name]
        else:
            # Fall back to single label
            result['real_labels'] = [result['label'].item()]
        
        return result
