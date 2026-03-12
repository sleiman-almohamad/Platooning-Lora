# Platooning-LoRA vs DoRA

A LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation) training project for generating platooning truck images using Stable Diffusion. This project compares standard LoRA and DoRA models to generate realistic images of truck platoons - convoys of trucks driving closely together on highways.

## 🚛 What is Platooning?

Platooning refers to a group of vehicles (typically trucks) that travel together in a convoy formation, maintaining close distances and coordinated movement. This technology is critical for autonomous vehicle research and fuel efficiency optimization.

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA RTX 4050 or better (tested on RTX 4050)
- **VRAM**: Minimum 8GB recommended
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Software Requirements
- **Python 3.10+** (Required for `lycoris-lora`)
- **uv**: Fast Python package manager
- CUDA-compatible PyTorch installation

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sleiman-almohamad/Platooning-Lora
cd Platooning-Lora
```

2. **Initialize the project:**
```bash
make init
```
This will install `uv`, create a virtual environment, install dependencies (including `lycoris-lora`), and download the base Stable Diffusion v1.5 model.

## 📁 Dataset Structure

Your dataset is organized by size to support grid search:

```
./dataset/
├── dataset_5/
├── dataset_10/
├── dataset_15/
├── dataset_25/
└── dataset_50/
```

Each subdirectory (e.g., `dataset_5`) contains a specific number of image-caption pairs for training.

## 🚀 Training

This project supports three training modes:

### 1. Standard LoRA Training
```bash
make train_lora
```
Uses [`training_pars_lora.toml`](training_pars_lora.toml) with standard LoRA settings.

### 2. DoRA Training
```bash
make train_dora
```
Uses [`training_pars_dora.toml`](training_pars_dora.toml) and the `lycoris.kohya` module with weight-decomposition (`dora_wd=true`).

### 3. Grid Search (Automated)
```bash
make grid-search
```
Runs a comprehensive grid search across:
- **Models**: LoRA, DoRA
- **Ranks**: 4, 8, 16, 32
- **Dataset Sizes**: 5, 10, 15, 25, 50

Outputs are organized in `grid_search_outputs/` with folders named like `lora_r32_data50`.

## 🔧 Project State

- **English Only**: The codebase has been cleaned of all non-English comments and strings for better maintainability.
- **Optimized for RTX 4050**: Performance and memory usage (approx. 7GB VRAM) have been tuned for midrange hardware.

## 🎨 Using the Trained Model

Use the generated `.safetensors` files with:
- **Automatic1111 WebUI**: Place in `models/Lora/`
- **ComfyUI**: Load via LoRA node
- **Diffusers**: `pipe.load_lora_weights("./path/to/model.safetensors")`

## 🧹 Cleanup

To clean up training files and start fresh:
```bash
make clean
```

---
**Acknowledgment**: Based on [sd-scripts](https://github.com/kohya-ss/sd-scripts) and [lycoris-lora](https://github.com/KohakuBlueLeaf/LyCORIS).