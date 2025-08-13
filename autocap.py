#!/usr/bin/env python3
"""
AutoCap - Automatic Image Captioning Tool for LoRA Training
Using Microsoft Florence-2-large model for high-quality image captioning
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import traceback
import gc

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autocap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CaptionMode(Enum):
    """Different captioning modes for various LoRA training purposes"""
    GENERAL = "general"
    STYLE = "style"
    CHARACTER = "character"
    OBJECT = "object"
    DETAILED = "detailed"
    SIMPLE = "simple"


class CaptionTask(Enum):
    """Florence-2 captioning tasks"""
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"


@dataclass
class CaptionConfig:
    """Configuration for captioning process"""
    mode: CaptionMode = CaptionMode.GENERAL
    task: CaptionTask = CaptionTask.DETAILED_CAPTION
    trigger_word: Optional[str] = None
    prepend_trigger: bool = True
    append_trigger: bool = False
    remove_objects: List[str] = field(default_factory=list)
    keep_style_words: bool = True
    max_length: int = 300
    min_length: int = 10
    batch_size: int = 1
    device: str = "auto"
    fp16: bool = True
    skip_existing: bool = True
    overwrite: bool = False


@dataclass
class ProcessingStats:
    """Statistics for processing session"""
    total_images: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> str:
        elapsed = datetime.now() - self.start_time
        return (
            f"\n{'='*50}\n"
            f"Processing Complete!\n"
            f"{'='*50}\n"
            f"Total Images: {self.total_images}\n"
            f"Processed: {self.processed}\n"
            f"Skipped (existing): {self.skipped}\n"
            f"Failed: {self.failed}\n"
            f"Time Elapsed: {elapsed}\n"
            f"{'='*50}"
        )


class Florence2Captioner:
    """Main class for Florence-2 based image captioning"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, config: CaptionConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = self._setup_device()
        self.stats = ProcessingStats()
        
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device for computation"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
            logger.info(f"Using specified device: {self.config.device}")
        
        return device
    
    def load_model(self):
        """Load Florence-2 model and processor"""
        try:
            logger.info("Loading Florence-2-large model...")
            
            model_id = "microsoft/Florence-2-large"
            
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use FP32 for CPU
                trust_remote_code=True
            )
            
            # Move model to the correct device
            self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
            if self.device.type == "cuda":
                self._log_gpu_memory()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_gpu_memory(self):
        """Log current GPU memory usage"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def process_image(self, image_path: Path) -> Optional[str]:
        """Process a single image and generate caption"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            prompt = self.config.task.value
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            if self.device.type != "cpu":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.config.fp16 and self.device.type != "cpu":
                    with torch.autocast(device_type=self.device.type):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            num_beams=3,
                            do_sample=False
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False
                    )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            parsed = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )
            
            caption = parsed[prompt]
            
            caption = self._process_caption(caption)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _process_caption(self, caption: str) -> str:
        """Process and clean the caption based on mode and configuration"""
        if not caption:
            return ""
        
        caption = caption.strip()
        
        if self.config.mode == CaptionMode.STYLE:
            caption = self._process_style_caption(caption)
        elif self.config.mode == CaptionMode.CHARACTER:
            caption = self._process_character_caption(caption)
        elif self.config.mode == CaptionMode.OBJECT:
            caption = self._process_object_caption(caption)
        elif self.config.mode == CaptionMode.SIMPLE:
            caption = self._simplify_caption(caption)
        
        if self.config.remove_objects:
            for obj in self.config.remove_objects:
                caption = caption.replace(obj, "").replace("  ", " ")
        
        if self.config.trigger_word:
            if self.config.prepend_trigger:
                caption = f"{self.config.trigger_word}, {caption}"
            elif self.config.append_trigger:
                caption = f"{caption}, {self.config.trigger_word}"
        
        if len(caption) > self.config.max_length:
            caption = self._truncate_caption(caption, self.config.max_length)
        
        caption = " ".join(caption.split())
        
        return caption
    
    def _process_style_caption(self, caption: str) -> str:
        """Process caption for style LoRA training"""
        style_keywords = [
            "artwork", "painting", "drawing", "illustration", "sketch",
            "digital art", "oil painting", "watercolor", "pencil drawing",
            "anime", "manga", "cartoon", "realistic", "photorealistic",
            "abstract", "surreal", "minimalist", "detailed", "vibrant",
            "monochrome", "colorful", "dark", "bright", "soft", "sharp"
        ]
        
        words = caption.lower().split()
        kept_words = []
        
        for word in caption.split():
            if any(keyword in word.lower() for keyword in style_keywords):
                kept_words.append(word)
            elif not any(obj in word.lower() for obj in ["person", "man", "woman", "people", "human"]):
                kept_words.append(word)
        
        return " ".join(kept_words)
    
    def _process_character_caption(self, caption: str) -> str:
        """Process caption for character LoRA training"""
        character_keywords = [
            "person", "man", "woman", "girl", "boy", "character",
            "face", "hair", "eyes", "clothing", "outfit", "pose",
            "expression", "standing", "sitting", "portrait"
        ]
        
        words = caption.split()
        enhanced_caption = caption
        
        has_character = any(keyword in caption.lower() for keyword in character_keywords)
        if not has_character:
            enhanced_caption = f"a character, {caption}"
        
        return enhanced_caption
    
    def _process_object_caption(self, caption: str) -> str:
        """Process caption for object LoRA training"""
        return caption.replace("a photo of", "").replace("an image of", "").strip()
    
    def _simplify_caption(self, caption: str) -> str:
        """Simplify caption to core elements"""
        remove_phrases = [
            "a photo of", "an image of", "a picture of",
            "this is", "there is", "there are",
            "showing", "featuring", "displaying"
        ]
        
        for phrase in remove_phrases:
            caption = caption.replace(phrase, "")
        
        return caption.strip()
    
    def _truncate_caption(self, caption: str, max_length: int) -> str:
        """Truncate caption intelligently at sentence or comma boundary"""
        if len(caption) <= max_length:
            return caption
        
        truncated = caption[:max_length]
        
        last_period = truncated.rfind('.')
        last_comma = truncated.rfind(',')
        
        if last_period > max_length * 0.8:
            return truncated[:last_period + 1]
        elif last_comma > max_length * 0.8:
            return truncated[:last_comma]
        else:
            last_space = truncated.rfind(' ')
            if last_space > 0:
                return truncated[:last_space] + "..."
            return truncated + "..."
    
    def process_directory(self, input_dir: Path, output_dir: Optional[Path] = None):
        """Process all images in a directory"""
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = self._get_image_files(input_dir)
        self.stats.total_images = len(image_files)
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        if not self.model:
            self.load_model()
        
        progress_bar = tqdm(image_files, desc="Processing images", unit="img")
        
        for image_path in progress_bar:
            caption_path = output_dir / f"{image_path.stem}.txt"
            
            if self.config.skip_existing and caption_path.exists() and not self.config.overwrite:
                logger.debug(f"Skipping {image_path.name} - caption already exists")
                self.stats.skipped += 1
                progress_bar.set_postfix({"skipped": self.stats.skipped})
                continue
            
            progress_bar.set_description(f"Processing {image_path.name}")
            
            caption = self.process_image(image_path)
            
            if caption:
                self._save_caption(caption, caption_path)
                self.stats.processed += 1
                progress_bar.set_postfix({"processed": self.stats.processed})
            else:
                self.stats.failed += 1
                progress_bar.set_postfix({"failed": self.stats.failed})
            
            if self.device.type == "cuda" and self.stats.processed % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        progress_bar.close()
        logger.info(self.stats.get_summary())
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all supported image files in directory"""
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    def _save_caption(self, caption: str, output_path: Path):
        """Save caption to text file"""
        try:
            output_path.write_text(caption, encoding='utf-8')
            logger.debug(f"Saved caption to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save caption to {output_path}: {e}")
    
    def process_batch(self, image_paths: List[Path]) -> List[Optional[str]]:
        """Process multiple images in batch (for future optimization)"""
        captions = []
        for image_path in image_paths:
            caption = self.process_image(image_path)
            captions.append(caption)
        return captions
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AutoCap - Automatic Image Captioning for LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python autocap.py /path/to/images
  
  Style LoRA with trigger word:
    python autocap.py /path/to/images --mode style --trigger "my_style"
  
  Character LoRA with detailed captions:
    python autocap.py /path/to/images --mode character --task MORE_DETAILED_CAPTION
  
  Process with custom output directory:
    python autocap.py /path/to/images --output /path/to/captions
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing images"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for caption files (default: same as input)"
    )
    
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=[mode.value for mode in CaptionMode],
        default=CaptionMode.GENERAL.value,
        help="Captioning mode for different LoRA types"
    )
    
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"],
        default="DETAILED_CAPTION",
        help="Florence-2 captioning task"
    )
    
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Trigger word to add to captions"
    )
    
    parser.add_argument(
        "--prepend-trigger",
        action="store_true",
        default=True,
        help="Add trigger word at the beginning of caption"
    )
    
    parser.add_argument(
        "--append-trigger",
        action="store_true",
        help="Add trigger word at the end of caption"
    )
    
    parser.add_argument(
        "--remove-objects",
        nargs="+",
        default=[],
        help="List of objects/words to remove from captions"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum caption length in characters"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum caption length in characters"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (currently supports 1)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision (use FP32)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing caption files"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Process images even if caption files exist"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Load configuration from JSON file"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save current configuration to JSON file"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def save_config_to_file(config: CaptionConfig, config_path: str):
    """Save configuration to JSON file"""
    try:
        config_dict = {
            "mode": config.mode.value,
            "task": config.task.value,
            "trigger_word": config.trigger_word,
            "prepend_trigger": config.prepend_trigger,
            "append_trigger": config.append_trigger,
            "remove_objects": config.remove_objects,
            "keep_style_words": config.keep_style_words,
            "max_length": config.max_length,
            "min_length": config.min_length,
            "batch_size": config.batch_size,
            "device": config.device,
            "fp16": config.fp16,
            "skip_existing": config.skip_existing,
            "overwrite": config.overwrite
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    config_dict = {}
    if args.config:
        config_dict = load_config_from_file(args.config)
    
    task_map = {
        "CAPTION": CaptionTask.CAPTION,
        "DETAILED_CAPTION": CaptionTask.DETAILED_CAPTION,
        "MORE_DETAILED_CAPTION": CaptionTask.MORE_DETAILED_CAPTION
    }
    
    config = CaptionConfig(
        mode=CaptionMode(config_dict.get("mode", args.mode)),
        task=task_map.get(config_dict.get("task", args.task), CaptionTask.DETAILED_CAPTION),
        trigger_word=config_dict.get("trigger_word", args.trigger),
        prepend_trigger=config_dict.get("prepend_trigger", args.prepend_trigger and not args.append_trigger),
        append_trigger=config_dict.get("append_trigger", args.append_trigger),
        remove_objects=config_dict.get("remove_objects", args.remove_objects),
        max_length=config_dict.get("max_length", args.max_length),
        min_length=config_dict.get("min_length", args.min_length),
        batch_size=config_dict.get("batch_size", args.batch_size),
        device=config_dict.get("device", args.device),
        fp16=config_dict.get("fp16", not args.no_fp16),
        skip_existing=config_dict.get("skip_existing", not args.no_skip),
        overwrite=config_dict.get("overwrite", args.overwrite)
    )
    
    if args.save_config:
        save_config_to_file(config, args.save_config)
    
    logger.info("="*50)
    logger.info("AutoCap - Image Captioning for LoRA Training")
    logger.info("="*50)
    logger.info(f"Mode: {config.mode.value}")
    logger.info(f"Task: {config.task.value}")
    if config.trigger_word:
        logger.info(f"Trigger Word: {config.trigger_word}")
    logger.info(f"Device: {config.device}")
    logger.info("="*50)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else None
    
    captioner = Florence2Captioner(config)
    
    try:
        captioner.process_directory(input_dir, output_dir)
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        captioner.cleanup()


if __name__ == "__main__":
    main()