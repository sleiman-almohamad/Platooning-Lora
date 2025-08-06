# Platooning-LoRA

A LoRA (Low-Rank Adaptation) training project for generating platooning truck images using Stable Diffusion. This project trains a custom LoRA model to generate realistic images of truck platoons - convoys of trucks driving closely together on highways.

## 🚛 What is Platooning?

Platooning refers to a group of vehicles (typically trucks) that travel together in a convoy formation, maintaining close distances and coordinated movement. This technology is important for autonomous vehicle research and fuel efficiency optimization.

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA RTX 4050 or better (tested on RTX 4050)
- **VRAM**: Minimum 8GB recommended
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Software Requirements
- Python 3.8+
- CUDA-compatible PyTorch installation
- Git

## 🛠️ Installation

### Windows

1. **Clone the repository:**
```bash
git clone https://github.com/sleiman-almohamad/Platooning-Lora
cd Platooning-Lora
```

2. **Install UV package manager (if not installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or download from: https://github.com/astral-sh/uv/releases

3. **Initialize the project:**
```bash
make init
```
This will:
- Create a virtual environment
- Install all dependencies
- Download the base Stable Diffusion model
- Verify the dataset

### Linux

1. **Clone the repository:**
```bash
git clone https://github.com/sleiman-almohamad/Platooning-Lora
cd Platooning-Lora
```

2. **Install UV package manager:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Initialize the project:**
```bash
make init
```

## 📁 Dataset Structure

Your dataset should be organized as follows:

```
./dataset/NumberOfRepetitions_images/
├── image1.png
├── image1.txt
├── image2.png
├── image2.txt
└── ...
```

**Important Notes:**
- Each image must have a corresponding `.txt` file with the same name
- The folder name format is: `NumberOfRepetitions_images` (e.g., `2_platooning` means each image will be repeated 2 times during training)
- Images should be in PNG or JPG format
- Text files should contain descriptive captions for the images

### Example Dataset Structure:
```
./dataset/2_platooning/
├── platooning1.png
├── platooning1.txt
├── platooning2.png
├── platooning2.txt
└── ...
```

## 🚀 Training

### Quick Start

1. **Prepare your dataset** in the correct folder structure (see above)

2. **Start training:**
```bash
make train
```

### Manual Training

If you prefer to run training manually:

```bash
# Windows
uv run train_network.py --config_file=training_pars.toml

# Linux
uv run train_network.py --config_file=training_pars.toml
```

### Training Configuration

The training parameters are configured in [`training_pars.toml`](training_pars.toml). Key settings include:

- **Epochs**: 25 (adjust `max_train_epochs`)
- **Learning Rate**: 5e-5 for U-Net, 1e-4 for Text Encoder
- **Batch Size**: 1 (increase if you have more VRAM)
- **Resolution**: 512x512
- **LoRA Rank**: 32 with alpha 16.0

### Hardware-Specific Notes

**Tested on RTX 4050:**
- Training time: ~2-3 hours for 25 epochs with 50 images
- Memory usage: ~6-7GB VRAM
- Recommended settings: Keep batch size at 1, enable gradient checkpointing


## 🎯 Sample Generation

The model generates sample images every epoch using prompts from the sample file. You can customize the prompts by editing the sample prompts file.

Example prompts:
- `platoontrucks, a long line of semi-trucks driving closely together on a highway, daytime, clear sky, realistic, highly detailed`
- `platoontrucks, a futuristic truck convoy, night, neon lights, cyberpunk city background`

## 📤 Output

Training outputs are saved to `Output_Lora_training/`:
- **LoRA weights**: `Lora_Platooning-000025.safetensors` (final model)
- **Intermediate checkpoints**: Saved every epoch
- **Sample images**: Generated during training
- **Training logs**: TensorBoard logs and training state

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `train_batch_size` to 1
- Enable `gradient_checkpointing = true`
- Reduce `network_dim` from 32 to 16

**Training Too Slow:**
- Increase `train_batch_size` if you have more VRAM
- Disable `cache_latents = false` if you have limited storage
- Reduce `max_train_epochs`

**Poor Quality Results:**
- Increase training epochs
- Improve dataset quality and captions
- Adjust learning rates
- Increase `network_dim` for more capacity

### Windows-Specific Issues

**UV not found:**
```bash
# Add UV to PATH or use full path
%USERPROFILE%\.cargo\bin\uv run train_network.py --config_file=training_pars.toml
```

**CUDA Issues:**
- Ensure CUDA toolkit is installed
- Verify PyTorch CUDA compatibility
- Check GPU drivers are up to date


**Missing Dependencies:**
```bash
sudo apt update
sudo apt install build-essential python3-dev
```

## 🎨 Using the Trained Model

After training, use the generated `.safetensors` file with:
- **Automatic1111 WebUI**: Place in `models/Lora/` folder
- **ComfyUI**: Load as LoRA node
- **Diffusers**: Load using `load_lora_weights()`

Recommended settings:
- **LoRA strength**: 0.7-1.0
- **Trigger word**: `platoontrucks`
- **CFG Scale**: 7-12
- **Steps**: 20-50

## 🧹 Cleanup

To clean up training files and start fresh:

```bash
make clean
```

This removes:
- Virtual environment
- Training outputs
- Cached files

## 📝 Configuration Reference

Key configuration options in [`training_pars.toml`](training_pars.toml):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_train_epochs` | Number of training epochs | 25 |
| `learning_rate` | Base learning rate | 5e-5 |
| `network_dim` | LoRA rank/dimension | 32 |
| `network_alpha` | LoRA alpha parameter | 16.0 |
| `train_batch_size` | Batch size per GPU | 1 |
| `resolution` | Training image resolution | 512,512 |
| `cache_latents` | Cache VAE latents to disk | true |

## 🙏 Acknowledgments

- Based on the excellent [sd-scripts](https://github.com/kohya-ss/sd-scripts) by kohya-ss
- Stable Diffusion model by Stability AI
- LoRA technique by Microsoft Research


**Hardware Note**: This project has been successfully tested and optimized for NVIDIA RTX 4050. Performance and memory usage may vary on different hardware configurations.