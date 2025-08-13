# AutoCap - AI-Powered Image Captioning for LoRA Training

**Created by Yuval Avidani**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/model-Florence--2--large-purple)](https://huggingface.co/microsoft/Florence-2-large)

AutoCap is a powerful, production-ready Python tool for automatically generating high-quality captions for image datasets using Microsoft's Florence-2-large vision-language model. Specifically designed for LoRA (Low-Rank Adaptation) training workflows, it provides intelligent captioning modes optimized for different training scenarios.

## 🌟 Features

- **🤖 Florence-2-large Model**: Leverages Microsoft's state-of-the-art vision-language model for accurate, detailed image understanding
- **🎯 Multiple Captioning Modes**: Specialized modes for style, character, object, and general LoRA training
- **⚡ Batch Processing**: Efficiently processes entire directories with smart resume capability
- **🏷️ Trigger Word Support**: Automatically prepends/appends trigger words for LoRA training
- **💾 Smart Caching**: Skip already processed images, with optional overwrite
- **🖥️ Multi-Device Support**: Automatic GPU/CPU detection with memory optimization
- **📊 Progress Tracking**: Real-time progress bar with detailed statistics
- **🔧 Highly Configurable**: JSON configuration support for saving and reusing settings
- **🛡️ Robust Error Handling**: Comprehensive error handling with detailed logging

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

### Basic Usage

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

## 📖 Detailed Usage

### Command Line Options

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