"""MS-COCO dataset for image-text retrieval evaluation."""

import os
import json
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torchvision import transforms

from .base import BaseEvalDataset


class COCORetrievalDataset(BaseEvalDataset):
    """
    MS-COCO dataset for image-text retrieval evaluation.
    
    This uses the Karpathy split which is standard for retrieval evaluation:
    - Train: 113,287 images
    - Val: 5,000 images  
    - Test: 5,000 images
    
    Each image has 5 captions.
    """
    
    name = "coco"
    task_type = "retrieval"
    
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        image_processor = None,
        max_samples: Optional[int] = None,
        use_restval: bool = False,
    ):
        """
        Initialize MS-COCO retrieval dataset.
        
        Args:
            data_root: Path to COCO directory (should contain images/ and annotations/)
            split: Dataset split ('train', 'val', 'test')
            image_processor: Image preprocessing function
            max_samples: Maximum number of samples (for debugging)
            use_restval: Whether to use restval split for training
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        self.use_restval = use_restval
        
        # Load samples
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded MS-COCO {split} set with {len(self.samples)} samples (retrieval)")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from COCO annotations."""
        samples = []
        
        # Try Karpathy split first (standard for retrieval)
        karpathy_file = os.path.join(self.data_root, "annotations", "coco_karpathy_split.json")
        if os.path.exists(karpathy_file):
            return self._load_karpathy_split(karpathy_file)
        
        # Fall back to standard COCO format
        return self._load_standard_coco()
    
    def _load_karpathy_split(self, karpathy_file: str) -> List[Dict[str, Any]]:
        """Load from Karpathy split JSON."""
        samples = []
        
        with open(karpathy_file, 'r') as f:
            data = json.load(f)
        
        # Map split names
        split_map = {
            'train': ['train', 'restval'] if self.use_restval else ['train'],
            'val': ['val'],
            'test': ['test'],
        }
        target_splits = split_map.get(self.split, [self.split])
        
        for item in data.get('images', []):
            if item.get('split') not in target_splits:
                continue
            
            # Get image path
            img_filename = item.get('filename', '')
            if not img_filename:
                continue
            
            # COCO images can be in train2014/ or val2014/
            image_path = None
            for subdir in ['train2014', 'val2014', 'images', '']:
                candidate = os.path.join(self.data_root, subdir, img_filename)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            # Get captions
            captions = []
            for sent in item.get('sentences', []):
                raw = sent.get('raw', sent.get('tokens', ''))
                if isinstance(raw, list):
                    raw = ' '.join(raw)
                if raw:
                    captions.append(raw)
            
            if not captions:
                continue
            
            # For retrieval, we use first caption (or all for multi-caption eval)
            samples.append({
                'image_path': image_path,
                'caption': captions[0],  # Primary caption
                'all_captions': captions,  # All 5 captions
                'image_id': item.get('imgid', item.get('cocoid', img_filename.split('.')[0])),
            })
        
        return samples
    
    def _load_standard_coco(self) -> List[Dict[str, Any]]:
        """Load from standard COCO annotation format."""
        samples = []
        
        # Standard COCO structure
        if self.split in ['val', 'test']:
            ann_file = os.path.join(self.data_root, "annotations", "captions_val2014.json")
            img_dir = os.path.join(self.data_root, "val2014")
        else:
            ann_file = os.path.join(self.data_root, "annotations", "captions_train2014.json")
            img_dir = os.path.join(self.data_root, "train2014")
        
        if not os.path.exists(ann_file):
            # Try alternative structure
            ann_file = os.path.join(self.data_root, "captions.json")
            img_dir = os.path.join(self.data_root, "images")
        
        if not os.path.exists(ann_file):
            print(f"Warning: COCO annotations not found at {ann_file}")
            return samples
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build image_id to filename mapping
        id_to_filename = {}
        for img in data.get('images', []):
            id_to_filename[img['id']] = img['file_name']
        
        # Group captions by image
        image_captions = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann['caption'])
        
        # Create samples
        for img_id, captions in image_captions.items():
            filename = id_to_filename.get(img_id)
            if not filename:
                continue
            
            image_path = os.path.join(img_dir, filename)
            if not os.path.exists(image_path):
                continue
            
            samples.append({
                'image_path': image_path,
                'caption': captions[0],
                'all_captions': captions,
                'image_id': img_id,
            })
        
        # Split for val/test if using standard format
        if self.split == 'test' and len(samples) > 5000:
            samples = samples[:5000]
        elif self.split == 'val' and len(samples) > 10000:
            samples = samples[5000:10000]
        
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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'caption': sample['caption'],
            'all_captions': sample.get('all_captions', [sample['caption']]),
            'image_id': sample['image_id'],
        }
    
    def get_all_captions(self) -> List[List[str]]:
        """Get all captions for each image (for multi-caption evaluation)."""
        return [s.get('all_captions', [s['caption']]) for s in self.samples]
