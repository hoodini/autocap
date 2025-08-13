#!/usr/bin/env python3
"""
AutoCap Enhanced - Smart trigger word integration for LoRA training
Enhanced version with better context understanding for trigger words
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Optional

# Import the main autocap module
from autocap import (
    Florence2Captioner, 
    CaptionConfig, 
    CaptionMode, 
    CaptionTask,
    ProcessingStats,
    logger
)


class EnhancedFlorence2Captioner(Florence2Captioner):
    """Enhanced captioner with smart trigger word integration"""
    
    def __init__(self, config: CaptionConfig, trigger_context: Optional[str] = None):
        super().__init__(config)
        self.trigger_context = trigger_context
    
    def _process_caption(self, caption: str) -> str:
        """Enhanced caption processing with context-aware trigger word integration"""
        if not caption:
            return ""
        
        caption = caption.strip()
        
        # Apply mode-specific processing first
        if self.config.mode == CaptionMode.STYLE:
            caption = self._process_style_caption(caption)
        elif self.config.mode == CaptionMode.CHARACTER:
            caption = self._process_character_caption(caption)
        elif self.config.mode == CaptionMode.OBJECT:
            caption = self._process_object_caption(caption)
        elif self.config.mode == CaptionMode.SIMPLE:
            caption = self._simplify_caption(caption)
        
        # Remove specified objects
        if self.config.remove_objects:
            for obj in self.config.remove_objects:
                caption = caption.replace(obj, "").replace("  ", " ")
        
        # Smart trigger word integration with context
        if self.config.trigger_word:
            caption = self._integrate_trigger_with_context(caption)
        
        # Length management
        if len(caption) > self.config.max_length:
            caption = self._truncate_caption(caption, self.config.max_length)
        
        caption = " ".join(caption.split())
        
        return caption
    
    def _integrate_trigger_with_context(self, caption: str) -> str:
        """Intelligently integrate trigger word with context understanding"""
        trigger = self.config.trigger_word
        
        # If we have context about what the trigger represents
        if self.trigger_context:
            # Check if the caption already describes something similar
            context_keywords = self.trigger_context.lower().split()
            caption_lower = caption.lower()
            
            # Find if any context keywords are in the caption
            has_context = any(keyword in caption_lower for keyword in context_keywords)
            
            if has_context:
                # Replace the generic term with our trigger
                for keyword in context_keywords:
                    if keyword in caption_lower:
                        # Smart replacement preserving case
                        caption = self._smart_replace(caption, keyword, trigger)
                        break
                else:
                    # If no direct replacement, prepend trigger
                    caption = f"{trigger}, {caption}"
            else:
                # Add trigger with context if the caption doesn't mention it
                if self.config.prepend_trigger:
                    caption = f"{trigger}, {caption}"
                elif self.config.append_trigger:
                    caption = f"{caption}, {trigger}"
        else:
            # Standard trigger word addition
            if self.config.prepend_trigger:
                caption = f"{trigger}, {caption}"
            elif self.config.append_trigger:
                caption = f"{caption}, {trigger}"
        
        return caption
    
    def _smart_replace(self, text: str, old_word: str, new_word: str) -> str:
        """Replace word preserving surrounding context"""
        import re
        # Case-insensitive replacement but preserve structure
        pattern = re.compile(re.escape(old_word), re.IGNORECASE)
        
        def replace_func(match):
            # If the matched text starts with uppercase, capitalize the replacement
            if match.group()[0].isupper():
                return new_word.capitalize() if new_word[0].islower() else new_word
            return new_word
        
        return pattern.sub(replace_func, text)


def quick_caption(
    folder_path: str,
    trigger_word: str,
    trigger_context: Optional[str] = None,
    mode: str = "object",
    task: str = "MORE_DETAILED_CAPTION"
):
    """
    Quick function to caption a folder with minimal configuration
    
    Args:
        folder_path: Path to folder containing images
        trigger_word: The trigger word for your LoRA (e.g., "yuvlabub")
        trigger_context: What the trigger represents (e.g., "labubu doll")
        mode: Captioning mode (object/character/style)
        task: Caption detail level
    """
    from pathlib import Path
    
    # Map string inputs to enums
    mode_map = {
        "object": CaptionMode.OBJECT,
        "character": CaptionMode.CHARACTER,
        "style": CaptionMode.STYLE,
        "general": CaptionMode.GENERAL,
        "detailed": CaptionMode.DETAILED,
        "simple": CaptionMode.SIMPLE
    }
    
    task_map = {
        "CAPTION": CaptionTask.CAPTION,
        "DETAILED_CAPTION": CaptionTask.DETAILED_CAPTION,
        "MORE_DETAILED_CAPTION": CaptionTask.MORE_DETAILED_CAPTION
    }
    
    config = CaptionConfig(
        mode=mode_map.get(mode, CaptionMode.OBJECT),
        task=task_map.get(task, CaptionTask.MORE_DETAILED_CAPTION),
        trigger_word=trigger_word,
        prepend_trigger=True,
        device="auto",
        fp16=True,
        skip_existing=True
    )
    
    captioner = EnhancedFlorence2Captioner(config, trigger_context)
    
    try:
        input_path = Path(folder_path)
        if not input_path.exists():
            print(f"Error: Folder {folder_path} does not exist!")
            return
        
        print(f"\n{'='*50}")
        print(f"Processing: {folder_path}")
        print(f"Trigger: {trigger_word}")
        if trigger_context:
            print(f"Context: {trigger_context}")
        print(f"Mode: {mode}")
        print(f"Task: {task}")
        print(f"{'='*50}\n")
        
        captioner.process_directory(input_path)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        captioner.cleanup()


def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(
        description="AutoCap Enhanced - Smart trigger word integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  For labubu dolls dataset:
    python autocap_enhanced.py "C:\\labubu_images" --trigger "yuvlabub" --context "labubu doll"
  
  For art style:
    python autocap_enhanced.py "C:\\art_images" --trigger "mystyle" --context "watercolor painting" --mode style
  
  For character:
    python autocap_enhanced.py "C:\\character_images" --trigger "john_doe" --context "man character" --mode character
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to folder containing images"
    )
    
    parser.add_argument(
        "--trigger",
        "-t",
        type=str,
        required=True,
        help="Trigger word for LoRA (e.g., 'yuvlabub')"
    )
    
    parser.add_argument(
        "--context",
        "-c",
        type=str,
        help="What the trigger represents (e.g., 'labubu doll')"
    )
    
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["object", "character", "style", "general", "detailed", "simple"],
        default="object",
        help="Captioning mode"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"],
        default="MORE_DETAILED_CAPTION",
        help="Caption detail level"
    )
    
    args = parser.parse_args()
    
    quick_caption(
        folder_path=args.input_dir,
        trigger_word=args.trigger,
        trigger_context=args.context,
        mode=args.mode,
        task=args.task
    )


if __name__ == "__main__":
    main()