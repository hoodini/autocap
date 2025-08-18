# AutoCap - AI-Powered Image Captioning for LoRA Training

**Created by Yuval Avidani**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/model-Florence--2--large-purple)](https://huggingface.co/microsoft/Florence-2-large)

AutoCap is a powerful, production-ready Python tool for automatically generating high-quality captions for image datasets using Microsoft's Florence-2-large vision-language model. Specifically designed for LoRA (Low-Rank Adaptation) training workflows, it provides intelligent captioning modes optimized for different training scenarios.

## 🌟 Features

- **🤖 Florence-2-large Model**: Leverages Microsoft's state-of-the-art vision-language model for accurate, detailed image understanding
- **🔢 Automatic Image Renaming**: Automatically rename images to sequential numbers (1.jpg, 2.jpg, etc.) for organized datasets
- **🎯 Multiple Captioning Modes**: Specialized modes for style, character, object, and general LoRA training
- **⚡ Batch Processing**: Efficiently processes entire directories with smart resume capability
- **🏷️ Advanced Trigger Word Support**: Full control over trigger sentences and trigger words for LoRA activation
- **💾 Smart Caching**: Skip already processed images, with optional overwrite
- **🖥️ Multi-Device Support**: Automatic GPU/CPU detection with memory optimization
- **📊 Progress Tracking**: Real-time progress bar with detailed statistics
- **🔧 Highly Configurable**: JSON configuration support for saving and reusing settings
- **🛡️ Robust Error Handling**: Comprehensive error handling with detailed logging
- **📦 Complete Dataset Preparation**: All-in-one tool for renaming, captioning, and adding trigger words

## 📋 Requirements

- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU optional but recommended for faster processing

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yuval-avidani/autocap.git
cd autocap
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: On first run, the script will automatically download the Florence-2-large model (~1.5GB).

## 💡 Quick Start

### Complete Dataset Preparation (Recommended)

Use `prepare_dataset.py` for the full workflow - automatic renaming, captioning, and trigger word setup:

```bash
# Character LoRA with complete automation
python prepare_dataset.py /path/to/photos \
    --trigger-sentence "a photo of person123" \
    --trigger-word "person123" \
    --mode character
```

### Basic Captioning Only

Caption all images in a folder with default settings:
```bash
python autocap.py /path/to/images
```

### LoRA Training Examples

**For Character/Object LoRA (e.g., Labubu dolls):**
```bash
python autocap.py "C:\labubu_images" --trigger "yuvlabub" --mode object --task MORE_DETAILED_CAPTION
```

**For Style LoRA:**
```bash
python autocap.py /path/to/art --trigger "mystyle" --mode style
```

**For Character LoRA:**
```bash
python autocap.py /path/to/characters --trigger "character_name" --mode character --task MORE_DETAILED_CAPTION
```

## 🎯 Complete Dataset Preparation Guide

### NEW: Understanding Trigger Words vs Trigger Sentences

**THIS IS CRUCIAL FOR LORA TRAINING SUCCESS!**

#### Trigger Sentence
- **What it is**: The complete phrase that starts every caption in your training files
- **Example**: `"a photo of johndoe person"`
- **Purpose**: Provides context and proper grammar for training

#### Trigger Word  
- **What it is**: The unique activation word you'll use after training
- **Example**: `"johndoe"`
- **Purpose**: The word you type in prompts to activate your LoRA

### Step-by-Step Workflow

#### 1. **Collect Your Images**
Place all images in a single folder. Any format (jpg, png, webp, etc.) works.

#### 2. **Run Complete Dataset Preparation** 
Use `prepare_dataset.py` - it handles everything automatically:

```bash
# Character LoRA Example
python prepare_dataset.py "./my_character_photos" \
    --trigger-sentence "a photo of alice123 person" \
    --trigger-word "alice123" \
    --mode character
```

**This automatically:**
- ✅ Renames all images to 1.jpg, 2.jpg, 3.jpg, etc.
- ✅ Generates detailed captions using Florence-2
- ✅ Adds your trigger sentence to the beginning of each caption
- ✅ Creates properly formatted .txt files for training

#### 3. **Result Example**
Your dataset will look like:
```
dataset/
├── 1.jpg
├── 1.txt ("a photo of alice123 person, smiling woman with brown hair wearing a red dress...")
├── 2.jpg  
├── 2.txt ("a photo of alice123 person, standing in a park wearing casual clothing...")
```

#### 4. **Start Training**
Use the trigger word `alice123` in your LoRA training configuration.

### Real-World Examples

#### Character LoRA (Person)
```bash
python prepare_dataset.py "./person_photos" \
    --trigger-sentence "a photo of uniquename person" \
    --trigger-word "uniquename" \
    --mode character
```
**After training, use**: `uniquename person wearing a suit` in your prompts

#### Art Style LoRA
```bash
python prepare_dataset.py "./artwork" \
    --trigger-sentence "artwork in style of mystyle" \
    --trigger-word "mystyle" \
    --mode style
```
**After training, use**: `mystyle, portrait of a woman` in your prompts

#### Object/Product LoRA
```bash
python prepare_dataset.py "./product_photos" \
    --trigger-sentence "a photo of productxyz item" \
    --trigger-word "productxyz" \
    --mode object
```
**After training, use**: `productxyz item on white background` in your prompts

### Important Rules for Trigger Words

1. **Make them unique**: Don't use common words like "person", "style", "art"
   - ✅ Good: `alice123`, `mystylexyz`, `productabc` 
   - ❌ Bad: `person`, `woman`, `style`

2. **Keep them simple**: Lowercase, no spaces or special characters
   - ✅ Good: `johndoe`, `mystylenew`, `uniquename`
   - ❌ Bad: `John-Doe`, `my style`, `unique_name`

3. **Be consistent**: Use the same trigger word throughout training and inference

### Advanced Options

#### Only Rename Images (No Captioning)
```bash
python prepare_dataset.py ./images --rename-only
```

#### Caption Without Renaming
```bash
python prepare_dataset.py ./images --no-rename
```

#### Add Trigger to Existing Captions
```bash
python prepare_dataset.py ./images \
    --no-autocap \
    --trigger-sentence "a photo of mysubject" \
    --trigger-word "mysubject"
```

#### Custom Starting Number for Renaming
```bash
python prepare_dataset.py ./images \
    --trigger-sentence "a photo of person123" \
    --trigger-word "person123" \
    --start-number 100
```
*Results in: 100.jpg, 101.jpg, 102.jpg, etc.*

## 📖 Detailed Usage

### prepare_dataset.py Options (Complete Dataset Preparation)

| Option | Description | Example |
|--------|-------------|---------|
| `dataset_path` | Path to the dataset directory | `./my_photos` |
| `--trigger-sentence` | Complete phrase to start each caption | `"a photo of alice123 person"` |
| `--trigger-word` | Unique activation word for LoRA | `"alice123"` |
| `--mode` | Caption mode: `character`, `style`, `object`, `detailed`, `simple` | `character` |
| `--rename-only` | Only rename images, skip captioning | |
| `--no-rename` | Skip image renaming step | |
| `--no-autocap` | Skip caption generation (for existing captions) | |
| `--start-number` | Starting number for sequential renaming | `1` |

### autocap.py Options (Captioning Only)

| Option | Description | Default |
|--------|-------------|---------|
| `input_dir` | Directory containing images to caption | Required |
| `--output, -o` | Output directory for caption files | Same as input |
| `--mode, -m` | Captioning mode: `general`, `style`, `character`, `object`, `detailed`, `simple` | `general` |
| `--task, -t` | Florence-2 task: `CAPTION`, `DETAILED_CAPTION`, `MORE_DETAILED_CAPTION` | `DETAILED_CAPTION` |
| `--trigger` | Trigger word to add to all captions (e.g., "yuvlabub") | None |
| `--prepend-trigger` | Add trigger word at beginning of caption | True |
| `--append-trigger` | Add trigger word at end of caption | False |
| `--remove-objects` | List of words to remove from captions | [] |
| `--max-length` | Maximum caption length in characters | 300 |
| `--device` | Device for inference: `auto`, `cuda`, `cpu`, `mps` | `auto` |
| `--no-fp16` | Disable FP16 precision (use FP32) | False |
| `--overwrite` | Overwrite existing caption files | False |
| `--no-skip` | Process images even if captions exist | False |
| `--config, -c` | Load configuration from JSON file | None |
| `--save-config` | Save current configuration to JSON file | None |
| `--verbose, -v` | Enable verbose logging | False |

### Captioning Modes Explained

#### 🎨 **Style Mode** (`--mode style`)
Optimized for art style LoRA training:
- Preserves artistic descriptors (watercolor, oil painting, digital art, etc.)
- Removes specific object references to focus on style
- Emphasizes style-related keywords

#### 👤 **Character Mode** (`--mode character`)
Optimized for character LoRA training:
- Emphasizes character features and attributes
- Includes pose and expression details
- Maintains character-specific descriptions

#### 📦 **Object Mode** (`--mode object`)
Optimized for object LoRA training:
- Removes unnecessary phrases ("a photo of", "an image of")
- Provides clean, direct object descriptions
- Perfect for product or specific item training

#### 📝 **General Mode** (`--mode general`)
Balanced captioning for general purpose training.

#### 🔍 **Detailed Mode** (`--mode detailed`)
Maximum detail captions using MORE_DETAILED_CAPTION task.

#### ⚡ **Simple Mode** (`--mode simple`)
Simplified captions with core elements only.

## 🎯 Real-World Example: Training a Labubu LoRA

```bash
# Process labubu doll images with trigger word
python autocap.py "C:\Users\User\Documents\ml-training\labubu" \
    --trigger "yuvlabub" \
    --mode object \
    --task MORE_DETAILED_CAPTION
```

**Output Example:**
- Image: `labubu_001.jpg`
- Caption: `yuvlabub, a white plush doll with rabbit ears and a round face, sitting on a wooden surface`

## 📁 Output Format

For each processed image, AutoCap creates a corresponding `.txt` file:
- `image_001.jpg` → `image_001.txt`
- `photo.png` → `photo.txt`

Each text file contains the generated caption with optional trigger word.

## ⚙️ Configuration Files

Save frequently used settings:
```bash
python autocap.py /path/to/images --mode style --trigger "mystyle" --save-config my_config.json
```

Load saved configuration:
```bash
python autocap.py /path/to/images --config my_config.json
```

Example configuration file:
```json
{
  "mode": "object",
  "task": "MORE_DETAILED_CAPTION",
  "trigger_word": "yuvlabub",
  "prepend_trigger": true,
  "remove_objects": [],
  "max_length": 300,
  "device": "auto",
  "fp16": false,
  "skip_existing": true
}
```

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- GIF (.gif)
- TIFF (.tiff)

## 🔧 Performance Optimization

### GPU Acceleration
The tool automatically detects and uses GPU if available:
- NVIDIA GPUs via CUDA
- Apple Silicon via MPS
- Falls back to CPU if no GPU available

### Memory Management
- Automatic GPU memory cleanup every 10 images
- FP16 precision support for lower memory usage
- Efficient batch processing architecture

### Processing Speed
- **GPU**: ~1-2 seconds per image
- **CPU**: ~4-9 seconds per image
- **Apple Silicon**: ~2-3 seconds per image

## 📊 Logging and Statistics

AutoCap provides comprehensive logging:
- Real-time progress bar with ETA
- Detailed statistics upon completion
- Log file (`autocap.log`) for debugging
- Use `--verbose` for detailed console output

## 🛠️ Troubleshooting

### Out of Memory Errors
```bash
# Disable FP16 for more stability
python autocap.py /path/to/images --no-fp16
```

### Slow Processing
- Ensure GPU is detected (check logs)
- First image takes longer due to model loading
- Consider using DETAILED_CAPTION instead of MORE_DETAILED_CAPTION

### Installation Issues
```bash
# For transformer compatibility issues
pip install transformers==4.45.0

# For missing dependencies
pip install einops timm
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Author

**Yuval Avidani**
- Creator and maintainer of AutoCap
- Specializing in AI/ML tools for creative workflows

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the Florence-2 model
- Hugging Face for the transformers library
- The LoRA training community for inspiration and use cases

## 📚 Citation

If you use AutoCap in your research or projects, please cite:
```
AutoCap - AI-Powered Image Captioning for LoRA Training
Created by Yuval Avidani
https://github.com/yuval-avidani/autocap
```

## 🚦 Project Status

**Active Development** - Regular updates and improvements

## 📮 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/yuval-avidani/autocap/issues)
- Check existing issues for solutions

---

**Made with ❤️ by Yuval Avidani for the AI art and LoRA training community**