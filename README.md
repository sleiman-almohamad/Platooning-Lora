# Platooning-LoRA: Enhanced Vehicle Recognition Training Pipeline

A sophisticated LoRA (Low-Rank Adaptation) training pipeline for generating platooning vehicle images using Stable Diffusion. This project enables fine-tuning of diffusion models to create high-quality images of vehicles in convoy formation.

## 🚀 Features

- **LoRA Fine-tuning**: Efficient parameter-efficient training using LoRA adapters
- **Stable Diffusion Integration**: Built on top of Stable Diffusion v1.5 architecture
- **Custom Dataset Support**: Designed for platooning vehicle image datasets
- **GPU Acceleration**: CUDA-optimized training with mixed precision support
- **Memory Efficient**: Gradient checkpointing and 8-bit Adam optimizer
- **Automated Pipeline**: Simple make commands for complete workflow

## 📋 Requirements

- **GPU**: CUDA-compatible GPU with at least 8GB VRAM
- **Python**: 3.9 or higher
- **Storage**: ~10GB free space for models and outputs
- **UV Package Manager**: For fast, reliable dependency management

### System Dependencies

```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# GPU drivers (NVIDIA)
nvidia-smi  # Verify GPU is detected
```

## 🛠️ Installation

The project uses **UV** for ultra-fast, reproducible dependency management with locked versions. 

### ✨ Why UV?
- **🚀 Speed**: 10-100x faster than pip for dependency resolution and installation
- **🔒 Reproducible**: Exact dependency versions locked in `uv.lock` (73 packages)
- **🛡️ Secure**: SHA256 checksums for all packages ensure integrity
- **🌍 Cross-platform**: Works consistently across Linux, macOS, and Windows
- **🔄 Reliable**: Never worry about "it works on my machine" problems again

The workflow includes essential commands:

### 1. Initialize Project
```bash
make init
```
This command:
- Creates a virtual environment (`.venv/`)
- Installs all dependencies from `uv.lock` and `pyproject.toml`
- Ensures reproducible, exact dependency versions
- Verifies the setup
- Checks for base model and dataset

### 2. Quick Training (Testing)
```bash
make train-quick
```
Runs a quick 10-epoch training session for testing purposes.

### 3. Full Training
```bash
make train
```
Runs full training with default 50 epochs. Customize with parameters:
```bash
make train EPOCHS=100 LEARNING_RATE=1e-5 BATCH_SIZE=2
```

### 4. Additional Commands
```bash
make info         # Show project and environment information
make update-deps  # Update all dependencies to latest versions
make add-dep PACKAGE=wandb  # Add new dependency and update lock file
make install      # Alias for init
```

### 5. Clean Up
```bash
make clean
```
Removes all generated files, outputs, virtual environment, and lock file.

## 📁 Project Structure

```
Platooning-Lora/
├── README.md                    # This file
├── Makefile                     # Build automation with UV integration
├── pyproject.toml              # Modern Python project configuration
├── uv.lock                     # Locked dependency versions (73 packages)
├── requirements_new.txt         # Legacy requirements (for reference)
├── run_training.py             # Training launcher script
├── train_new.py                # Enhanced training pipeline
├── inference_new.py            # Inference utilities
├── v1-5-pruned.safetensors     # Base Stable Diffusion model (7.2GB)
├── Platooning images-resized/  # Training dataset (50 images)
│   ├── image_001.png
│   ├── image_001.txt           # Caption for image_001.png
│   └── ...
├── .venv/                      # Virtual environment (created by make init)
└── platooning_lora_v2/         # Training outputs (created during training)
    ├── checkpoint-best/        # Best model checkpoint
    ├── checkpoint-final/       # Final model checkpoint
    └── training_config.json    # Training configuration
```

## 📊 Dataset Format

The training dataset should contain paired image and text files:

- **Images**: PNG format, 512x512 resolution recommended
- **Captions**: Text files with the same name as images
- **Format**: `image_001.png` paired with `image_001.txt`

### Example Caption Format:
```
Two trucks platooning on highway in convoy formation
```

## ⚙️ Training Configuration

The training pipeline uses these default parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 50 | Number of training epochs |
| `BATCH_SIZE` | 1 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `LORA_RANK` | 16 | LoRA rank parameter |

### Advanced Configuration

For advanced users, modify the configuration in `train_new.py`:

```python
class Config:
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training settings
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = True
```

## 🎯 Usage Examples

### Basic Workflow
```bash
# 1. Set up the project (installs from uv.lock)
make init

# 2. Check project status
make info

# 3. Test with quick training
make train-quick

# 4. Run full training
make train

# 5. Clean up when done
make clean
```

### Custom Training Parameters
```bash
# Long training with small learning rate
make train EPOCHS=200 LEARNING_RATE=5e-5

# Larger batch size (if you have enough VRAM)
make train BATCH_SIZE=4

# Quick test with different parameters
make train-quick BATCH_SIZE=2 LORA_RANK=32
```

### Dependency Management
```bash
# Add new dependencies
make add-dep PACKAGE=wandb           # Add Weights & Biases logging
make add-dep PACKAGE=tensorboard     # Add TensorBoard support

# Update all dependencies to latest versions
make update-deps

# Show project information and dependency status
make info
```

## 📈 Training Monitoring

### Training Progress
The training script provides real-time progress updates:
- Loss curves and learning rate schedules
- GPU memory usage monitoring
- Automatic checkpointing every 10 epochs
- Validation loss tracking

### Output Structure
```
platooning_lora_v2/
├── checkpoint-best/           # Best performing model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_config.json
├── checkpoint-final/          # Final epoch model
└── checkpoint-{epoch}/        # Periodic checkpoints
```

## 🔧 Troubleshooting

### Common Issues

**GPU Out of Memory**
```bash
# Reduce batch size
make train BATCH_SIZE=1

# Or reduce image resolution in train_new.py
image_size: int = 256  # Instead of 512
```

**Dependencies Issues**
```bash
# Reinitialize environment
make clean
make init
```

**Training Fails to Start**
```bash
# Check if base model exists
ls -la v1-5-pruned.safetensors

# Verify dataset
ls "Platooning images-resized"/*.png | wc -l
```

### Performance Optimization

**For Better Performance:**
- Use batch size 2-4 if you have >12GB VRAM
- Enable gradient checkpointing for memory efficiency
- Use mixed precision (fp16) training
- Consider using xformers for attention optimization

**For Stability:**
- Use batch size 1 for limited VRAM
- Reduce learning rate for sensitive datasets
- Increase gradient accumulation steps

## 📚 Technical Details

### Model Architecture
- **Base Model**: Stable Diffusion v1.5
- **Adaptation**: LoRA (Low-Rank Adaptation)
- **Target Modules**: Attention and feed-forward layers
- **Precision**: Mixed precision (FP16/FP32)

### Training Features
- **Noise Offset**: Improved color range handling
- **Data Augmentation**: Random horizontal flip, color jitter
- **Validation Split**: 80/20 train/validation split
- **Scheduler**: Cosine annealing learning rate schedule
- **Optimizer**: 8-bit AdamW for memory efficiency

### Hardware Requirements
- **Minimum**: 8GB VRAM (RTX 3070/4060 Ti)
- **Recommended**: 12GB+ VRAM (RTX 3080/4070 Ti/4080)
- **Optimal**: 24GB+ VRAM (RTX 4090/A6000)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make train-quick`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Stable Diffusion by Stability AI
- LoRA by Microsoft Research
- Diffusers library by Hugging Face
- PEFT library for parameter-efficient fine-tuning

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the training logs for error details
3. Ensure your GPU meets the minimum requirements
4. Verify dataset format and file structure

---

**Happy Training! 🚛🚛🚛**
