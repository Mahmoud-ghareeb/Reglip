#!/usr/bin/env python3
"""
Standalone script to validate the Flickr30K dataset.

Usage:
    python validate_dataset.py
    python validate_dataset.py --data-root /path/to/flickr30k
    python validate_dataset.py --quiet  # Less verbose output
"""

import argparse
import sys
from data.preprocessing import validate_flickr30k_dataset, quick_data_check


def main():
    parser = argparse.ArgumentParser(description="Validate Flickr30K dataset")
    parser.add_argument(
        "--data-root", 
        type=str, 
        default="/home/mahmoud/RegLIP/data/flickr30k",
        help="Path to Flickr30K dataset directory"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Less verbose output"
    )
    parser.add_argument(
        "--quick-only", 
        action="store_true",
        help="Only run quick check"
    )
    
    args = parser.parse_args()
    
    print("🔍 FLICKR30K DATASET VALIDATOR")
    print("=" * 50)
    print(f"Dataset path: {args.data_root}")
    print()
    
    if args.quick_only:
        # Quick check only
        print("Running quick check...")
        if quick_data_check(args.data_root):
            print("✅ Quick check passed - basic structure exists")
            return 0
        else:
            print("❌ Quick check failed - dataset structure incomplete")
            return 1
    
    # Full validation
    verbose = not args.quiet
    results = validate_flickr30k_dataset(args.data_root, verbose=verbose)
    
    if not verbose:
        # Print summary even in quiet mode
        print("\nSUMMARY:")
        if results['valid']:
            print("✅ Dataset validation passed")
            if results['summary']:
                summary = results['summary']
                print(f"📊 Usable samples: {summary['total_usable_samples']}")
                splits = summary['estimated_splits']
                print(f"📊 Estimated splits - Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}")
        else:
            print("❌ Dataset validation failed")
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
    
    # Return appropriate exit code
    return 0 if results['valid'] else 1


if __name__ == "__main__":
    sys.exit(main()) 