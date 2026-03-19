"""Flickr30K dataset for retrieval evaluation."""

import os
from typing import Dict, Any, Optional, List
from PIL import Image
import torch

from .base import BaseEvalDataset


class Flickr30KRetrievalDataset(BaseEvalDataset):
    """Flickr30K dataset for image-text retrieval evaluation."""
    
    name = "flickr30k"
    task_type = "retrieval"
    
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        image_processor = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize Flickr30K retrieval dataset.
        
        Args:
            data_root: Path to flickr30k directory
            split: Dataset split ('train', 'val', 'test')
            image_processor: Image preprocessing function
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = data_root
        self.split = split
        self.image_processor = image_processor
        
        # Load annotations
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples for {split} split (retrieval)")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from captions file."""
        annotation_file = os.path.join(self.data_root, "captions.txt")
        images_dir = os.path.join(self.data_root, "Images")
        
        samples = []
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            # Group captions by image (take first caption for each image for retrieval)
            seen_images = set()
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',', 1)  # Split on first comma only
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        caption = parts[1].strip()
                        
                        # Skip if already seen this image or missing data
                        if filename in seen_images or not filename or not caption:
                            continue
                        
                        # Check if image exists
                        image_path = os.path.join(images_dir, filename)
                        if not os.path.exists(image_path):
                            continue
                        
                        seen_images.add(filename)
                        samples.append({
                            'image_path': image_path,
                            'caption': caption,
                            'image_id': filename.split('.')[0],
                        })
        
        # Apply split
        if self.split in ['train', 'val', 'test'] and len(samples) > 100:
            total = len(samples)
            if self.split == 'train':
                samples = samples[:int(0.8 * total)]
            elif self.split == 'val':
                samples = samples[int(0.8 * total):int(0.9 * total)]
            else:  # test
                samples = samples[int(0.9 * total):]
        
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
            pixel_values = image
        
        return {
            'pixel_values': pixel_values,
            'caption': sample['caption'],
            'image_id': sample['image_id'],
        }
