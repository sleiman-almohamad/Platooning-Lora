#!/usr/bin/env python3
"""
Enhanced LoRA Training Pipeline for Platooning Vehicle Recognition
Redesigned from scratch with improved architecture and training procedures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime

# Diffusion and LoRA imports
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
from accelerate import Accelerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Training configuration with improved defaults"""
    
    # Data settings
    data_dir: str = "./Platooning images-resized"
    image_size: int = 512  # Full resolution for better quality
    batch_size: int = 1    # Small batch for memory efficiency
    num_workers: int = 2
    
    # Model settings
    model_id: str = "v1-5-pruned.safetensors"
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    
    # LoRA settings
    lora_rank: int = 16        # Increased rank for better capacity
    lora_alpha: int = 32       # Higher alpha for stronger adaptation
    lora_dropout: float = 0.1
    target_modules: List[str] = [
        "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
        "ff.net.0.proj", "ff.net.2"          # Feed-forward layers
    ]
    
    # Training settings
    learning_rate: float = 1e-4
    num_epochs: int = 50       # More epochs for better convergence
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    noise_offset: float = 0.1  # Improved noise handling
    
    # Scheduler settings
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    
    # Output settings
    output_dir: str = "platooning_lora_v2"
    save_every_n_epochs: int = 10
    validation_prompts: List[str] = [
        "Two trucks platooning on highway",
        "Autonomous vehicles in convoy formation",
        "Commercial trucks following in close formation",
        "Platooning vehicles maintaining safe distance"
    ]
    
    # Logging
    use_wandb: bool = False
    project_name: str = "platooning-lora"
    
    # Advanced settings
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True


class EnhancedPlatooningDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    
    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split
        
        # Find all image files
        self.image_paths = sorted(glob.glob(os.path.join(config.data_dir, "*.png")))
        logger.info(f"Found {len(self.image_paths)} images in {config.data_dir}")
        
        # Split dataset (80% train, 20% validation)
        split_idx = int(0.8 * len(self.image_paths))
        if split == "train":
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
        
        logger.info(f"{split} split: {len(self.image_paths)} images")
        
        # Enhanced image transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and process image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image_tensor = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        # Load text description
        text_path = image_path.replace(".png", ".txt")
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    # Clean up text format (remove numbering if present)
                    if '|' in text:
                        text = text.split('|', 1)[1].strip()
            except Exception as e:
                logger.warning(f"Error loading text {text_path}: {e}")
                text = "platooning vehicles on road"
        else:
            text = "platooning vehicles on road"
        
        return {
            "pixel_values": image_tensor,
            "input_ids": text,
            "image_path": image_path
        }


class PlatooningLoRATrainer:
    """Enhanced trainer with better architecture and monitoring"""
    
    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project=config.project_name, config=vars(config))
        
        self.setup_models()
        self.setup_data()
        self.setup_training()
    
    def setup_models(self):
        """Setup and configure all models"""
        logger.info("Setting up models...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name, 
            subfolder="text_encoder"
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name, 
            subfolder="vae"
        )
        
        # Load UNet
        if os.path.exists(self.config.model_id):
            logger.info(f"Loading UNet from local file: {self.config.model_id}")
            # Load from safetensors file
            pipeline = StableDiffusionPipeline.from_single_file(
                self.config.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            self.unet = pipeline.unet
        else:
            logger.info(f"Loading UNet from HuggingFace: {self.config.pretrained_model_name}")
            self.unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_model_name, 
                subfolder="unet"
            )
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name, 
            subfolder="scheduler"
        )
        
        # Freeze models except UNet
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        # Apply LoRA to UNet
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Move models to device
        self.vae.to(self.accelerator.device, dtype=torch.float16)
        self.text_encoder.to(self.accelerator.device, dtype=torch.float16)
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        logger.info("Setting up datasets...")
        
        self.train_dataset = EnhancedPlatooningDataset(self.config, split="train")
        self.val_dataset = EnhancedPlatooningDataset(self.config, split="val")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        logger.info("Setting up training components...")
        
        # Setup optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("bitsandbytes not available, using regular AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW
        
        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # Setup scheduler
        if self.config.lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs * len(self.train_dataloader)
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=10, 
                gamma=0.5
            )
        
        # Prepare for distributed training
        self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.accelerator.device))[0]
        
        return text_embeddings
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        with self.accelerator.accumulate(self.unet):
            # Convert images to latent space
            latents = self.vae.encode(batch["pixel_values"].to(dtype=self.vae.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            if self.config.noise_offset > 0:
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=latents.device
            ).long()
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            text_embeddings = self.encode_text(batch["input_ids"])
            
            # Predict noise
            model_pred = self.unet(noisy_latents, timesteps, text_embeddings, return_dict=False)[0]
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Backward pass
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.detach().item()
    
    def validate(self) -> float:
        """Validation step"""
        self.unet.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Convert images to latent space
                latents = self.vae.encode(batch["pixel_values"].to(dtype=self.vae.dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                text_embeddings = self.encode_text(batch["input_ids"])
                
                # Predict noise
                model_pred = self.unet(noisy_latents, timesteps, text_embeddings, return_dict=False)[0]
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_loss += loss.item()
                num_batches += 1
        
        self.unet.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_model(self, epoch: int):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        self.accelerator.save_state(save_path)
        
        # Save LoRA weights
        if self.accelerator.is_main_process:
            self.unet.save_pretrained(save_path)
            
            # Save config
            config_dict = vars(self.config)
            with open(os.path.join(save_path, "training_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Model saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.unet.train()
            total_loss = 0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                total_loss += loss
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{total_loss / (step + 1):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to wandb
                if self.config.use_wandb and self.accelerator.is_main_process:
                    wandb.log({
                        "train_loss": loss,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "global_step": global_step
                    })
            
            # Validation
            val_loss = self.validate()
            avg_train_loss = total_loss / len(self.train_dataloader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log validation metrics
            if self.config.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "val_loss": val_loss,
                    "avg_train_loss": avg_train_loss,
                    "epoch": epoch
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_model(epoch + 1)
        
        # Save final model
        self.save_model("final")
        logger.info("Training completed!")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Enhanced LoRA Training for Platooning Recognition")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./Platooning images-resized", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="platooning_lora_v2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.lora_rank:
        config.lora_rank = args.lora_rank
    if args.use_wandb:
        config.use_wandb = True
    
    # Initialize trainer and start training
    trainer = PlatooningLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
