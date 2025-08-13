#!/usr/bin/env python3
"""
Simple example showing how to use AutoCap for your specific use case
"""

from pathlib import Path
from autocap import Florence2Captioner, CaptionConfig, CaptionMode, CaptionTask

def caption_my_lora_dataset(folder_path: str, trigger_word: str):
    """
    Simple function to caption your LoRA dataset
    
    Example usage:
        caption_my_lora_dataset("C:\\labubu_images", "yuvlabub")
    """
    
    # Configure for your needs
    config = CaptionConfig(
        mode=CaptionMode.OBJECT,  # Use OBJECT mode for dolls/objects
        task=CaptionTask.MORE_DETAILED_CAPTION,  # Get detailed descriptions
        trigger_word=trigger_word,  # Your LoRA trigger
        prepend_trigger=True,  # Add trigger at the beginning
        device="auto",  # Auto-detect GPU
        fp16=True,  # Use FP16 for faster processing
        skip_existing=True  # Skip if caption already exists
    )
    
    # Create captioner
    captioner = Florence2Captioner(config)
    
    # Process all images in the folder
    input_path = Path(folder_path)
    print(f"\nProcessing images in: {folder_path}")
    print(f"Trigger word: {trigger_word}")
    print("-" * 50)
    
    captioner.process_directory(input_path)
    
    # Clean up
    captioner.cleanup()
    
    print("\nDone! Check your folder for .txt files with captions.")


if __name__ == "__main__":
    # Example: Caption labubu dolls dataset
    # Just change these two values for your dataset:
    
    FOLDER = "C:\\path\\to\\your\\images"  # <-- Change this to your folder path
    TRIGGER = "yuvlabub"  # <-- Change this to your trigger word
    
    caption_my_lora_dataset(FOLDER, TRIGGER)