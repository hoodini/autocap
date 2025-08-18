#!/usr/bin/env python3
"""
Dataset Preparation Tool for LoRA Training
Automatically renames images, generates captions, and adds trigger words
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional

def rename_images_sequential(dataset_path: str, start_number: int = 1) -> int:
    """
    Rename all images in the dataset to sequential numbers.
    Returns the number of images renamed.
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_path} does not exist")
        return 0
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
    image_files = []
    
    for item in dataset_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            # Skip if already numbered
            if not item.stem.isdigit():
                image_files.append(item)
    
    if not image_files:
        print("No images to rename (all already numbered or no images found)")
        return 0
    
    # Sort files for consistent ordering
    image_files.sort()
    
    print(f"Renaming {len(image_files)} images...")
    
    # First, rename to temporary names to avoid conflicts
    temp_renames = []
    for idx, old_file in enumerate(image_files):
        temp_name = dataset_dir / f"temp_{idx}_{old_file.name}"
        shutil.move(str(old_file), str(temp_name))
        temp_renames.append(temp_name)
    
    # Now rename to final sequential names
    for idx, temp_file in enumerate(temp_renames, start=start_number):
        ext = temp_file.suffix.lower()
        new_filename = f"{idx}{ext}"
        new_filepath = dataset_dir / new_filename
        shutil.move(str(temp_file), str(new_filepath))
        print(f"  Renamed to: {new_filename}")
    
    return len(image_files)

def add_trigger_to_captions(dataset_path: str, trigger_sentence: str) -> int:
    """
    Add trigger sentence to the beginning of all caption files.
    Returns the number of files updated.
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_path} does not exist")
        return 0
    
    caption_files = list(dataset_dir.glob('*.txt'))
    
    if not caption_files:
        print("No caption files found")
        return 0
    
    print(f"Adding trigger sentence to {len(caption_files)} caption files...")
    updated_count = 0
    
    for caption_file in caption_files:
        with open(caption_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Check if already has the trigger sentence
        if content.startswith(trigger_sentence):
            continue
        
        # Add trigger sentence
        new_content = f"{trigger_sentence}, {content}" if content else trigger_sentence
        
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        updated_count += 1
    
    print(f"  Updated {updated_count} files")
    return updated_count

def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for LoRA training with automatic renaming and trigger words',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just rename images to sequential numbers
  python prepare_dataset.py /path/to/dataset --rename-only
  
  # Rename and run autocap with trigger word
  python prepare_dataset.py /path/to/dataset --trigger-sentence "a woman in style of artname" --trigger-word "artname"
  
  # Run autocap and add trigger sentence (no renaming)
  python prepare_dataset.py /path/to/dataset --no-rename --trigger-sentence "a man yuvalavyuvai" --trigger-word "yuvalavyuvai"
  
  # Full pipeline for character LoRA
  python prepare_dataset.py /path/to/dataset --trigger-sentence "a photo of person123" --trigger-word "person123" --mode character
        """
    )
    
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('--rename-only', action='store_true', 
                       help='Only rename images, do not run autocap')
    parser.add_argument('--no-rename', action='store_true',
                       help='Skip image renaming step')
    parser.add_argument('--trigger-sentence', type=str,
                       help='Complete sentence to prepend to each caption (e.g., "a man yuvalavyuvai")')
    parser.add_argument('--trigger-word', type=str,
                       help='Single trigger word for training (e.g., "yuvalavyuvai")')
    parser.add_argument('--mode', type=str, default='detailed',
                       choices=['general', 'style', 'character', 'object', 'detailed', 'simple'],
                       help='Captioning mode for autocap (default: detailed)')
    parser.add_argument('--start-number', type=int, default=1,
                       help='Starting number for sequential renaming (default: 1)')
    parser.add_argument('--no-autocap', action='store_true',
                       help='Skip autocap generation (useful if captions already exist)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_path}' does not exist")
        sys.exit(1)
    
    print("="*60)
    print("Dataset Preparation for LoRA Training")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    
    # Step 1: Rename images if requested
    if not args.no_rename:
        print("\n► Step 1: Renaming images to sequential numbers...")
        renamed_count = rename_images_sequential(str(dataset_path), args.start_number)
        if renamed_count > 0:
            print(f"✓ Renamed {renamed_count} images")
        else:
            print("✓ No images needed renaming")
    else:
        print("\n► Step 1: Skipping image renaming")
    
    if args.rename_only:
        print("\n✓ Rename-only mode - complete!")
        return
    
    # Step 2: Run autocap if not skipped
    if not args.no_autocap:
        print("\n► Step 2: Generating captions with AutoCap...")
        
        # Get the autocap.py path (assuming it's in the same directory)
        script_dir = Path(__file__).parent
        autocap_path = script_dir / "autocap.py"
        
        if not autocap_path.exists():
            print(f"Error: autocap.py not found at {autocap_path}")
            sys.exit(1)
        
        # Build autocap command
        cmd = f'python "{autocap_path}" "{dataset_path}" --mode {args.mode} --overwrite'
        
        print(f"Running: {cmd}")
        result = os.system(cmd)
        
        if result != 0:
            print("Error: AutoCap failed")
            sys.exit(1)
        
        print("✓ Captions generated successfully")
    else:
        print("\n► Step 2: Skipping AutoCap generation")
    
    # Step 3: Add trigger sentence if provided
    if args.trigger_sentence:
        print(f"\n► Step 3: Adding trigger sentence to captions...")
        print(f"  Trigger sentence: '{args.trigger_sentence}'")
        if args.trigger_word:
            print(f"  Trigger word: '{args.trigger_word}'")
        
        updated_count = add_trigger_to_captions(str(dataset_path), args.trigger_sentence)
        print(f"✓ Updated {updated_count} caption files")
    else:
        print("\n► Step 3: No trigger sentence provided - skipping")
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    
    if args.trigger_sentence and args.trigger_word:
        print(f"\n📝 Training Configuration:")
        print(f"  • Trigger word for training: {args.trigger_word}")
        print(f"  • Each caption starts with: {args.trigger_sentence}")
        print(f"\n💡 Tips:")
        print(f"  • Use '{args.trigger_word}' as the activation word in your LoRA training config")
        print(f"  • During inference, use '{args.trigger_word}' to activate your trained style/character")

if __name__ == "__main__":
    main()