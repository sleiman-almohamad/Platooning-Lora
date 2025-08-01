#!/usr/bin/env python3
"""
Training launcher script with environment setup and validation
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'diffusers', 'transformers', 
        'peft', 'accelerate', 'safetensors', 'PIL', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install with: pip install -r requirements_new.txt")
        return False
    
    return True

def check_dataset():
    """Check if dataset is properly formatted"""
    data_dir = Path("./Platooning _images-resized")
    
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return False
    
    png_files = list(data_dir.glob("*.png"))
    txt_files = list(data_dir.glob("*.txt"))
    
    logger.info(f"Found {len(png_files)} PNG files and {len(txt_files)} text files")
    
    if len(png_files) == 0:
        logger.error("No PNG files found in dataset")
        return False
    
    # Check if text files exist for images
    missing_txt = []
    for png_file in png_files[:5]:  # Check first 5
        txt_file = png_file.with_suffix('.txt')
        if not txt_file.exists():
            missing_txt.append(str(txt_file))
    
    if missing_txt:
        logger.warning(f"Some text files missing: {missing_txt[:3]}...")
    
    return True

def check_base_model():
    """Check if base model exists"""
    model_path = Path("v1-5-pruned.safetensors")
    
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        logger.info(f"Base model found: {model_path} ({size_gb:.1f} GB)")
        return True
    else:
        logger.warning("Base model v1-5-pruned.safetensors not found")
        logger.info("Training will download from HuggingFace (slower)")
        return True  # Still allow training

def run_training(args=None):
    """Run the training script with proper setup"""
    
    logger.info("=== Enhanced LoRA Training Setup ===")
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Check dataset
    if not check_dataset():
        return False
    
    # Check base model
    check_base_model()
    
    # Prepare training command
    cmd = [sys.executable, "train_new.py"]
    
    if args:
        cmd.extend(args)
    
    logger.info(f"Starting training: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        logger.info("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return False

if __name__ == "__main__":
    # Parse command line arguments for training script
    training_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    success = run_training(training_args)
    sys.exit(0 if success else 1)
