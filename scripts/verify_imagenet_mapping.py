#!/usr/bin/env python3
"""
Utility script to verify or create ImageNet folder-to-class mapping.

This helps diagnose label mismatches when using numbered folders (00000-00999).
"""

import os
import json
import argparse
from PIL import Image


def load_imagenet_classes(data_root: str):
    """Load ImageNet class names."""
    classes_file = os.path.join(data_root, "imagenet_classes.json")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return json.load(f)
    return None


def verify_mapping(data_root: str, num_samples: int = 5):
    """
    Verify if numbered folders match class indices.
    
    Args:
        data_root: Path to ImageNet directory
        num_samples: Number of folders to check
    """
    classes = load_imagenet_classes(data_root)
    if not classes:
        print(f"Error: Could not find imagenet_classes.json in {data_root}")
        return
    
    print(f"Loaded {len(classes)} ImageNet classes")
    print(f"\nVerifying folder-to-class mapping:")
    print("=" * 80)
    
    # Check first N and last N folders
    folders_to_check = []
    for i in range(min(num_samples, 10)):
        folders_to_check.append(f"{i:05d}")
    for i in range(max(0, 1000 - num_samples), 1000):
        folders_to_check.append(f"{i:05d}")
    
    mismatches = []
    for folder_num in folders_to_check:
        folder_path = os.path.join(data_root, folder_num)
        if not os.path.exists(folder_path):
            continue
        
        folder_idx = int(folder_num)
        expected_class = classes[folder_idx] if folder_idx < len(classes) else "N/A"
        
        images = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Folder {folder_num} -> Class index {folder_idx}: {expected_class}")
        print(f"  Images: {len(images)}")
        
        # Visual check - user should verify if images match the class name
        if images:
            sample_img = os.path.join(folder_path, images[0])
            try:
                img = Image.open(sample_img)
                print(f"  Sample: {images[0]} ({img.size})")
                print(f"  ⚠️  Please verify: Do these images look like '{expected_class}'?")
            except Exception as e:
                print(f"  Error loading image: {e}")
        print()
    
    print("=" * 80)
    print("\nIf the images don't match the expected class names, you need a mapping file.")
    print("Create 'folder_to_class_mapping.json' in the data root with format:")
    print('  {"0": 123, "1": 456, ...}  # Maps folder number to class index')


def create_mapping_template(data_root: str, output_file: str = "folder_to_class_mapping.json"):
    """Create a template mapping file."""
    mapping = {}
    for i in range(1000):
        mapping[str(i)] = i  # Default: folder number = class index
    
    output_path = os.path.join(data_root, output_file)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Created mapping template at {output_path}")
    print("Edit this file to fix any mismatches between folder numbers and class indices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ImageNet folder-to-class mapping")
    parser.add_argument("data_root", help="Path to ImageNet directory")
    parser.add_argument("--samples", type=int, default=5, 
                       help="Number of folders to check (default: 5)")
    parser.add_argument("--create-template", action="store_true",
                       help="Create a template mapping file")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_mapping_template(args.data_root)
    else:
        verify_mapping(args.data_root, args.samples)
