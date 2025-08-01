import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import json
import toml
import logging
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, List

# Diffusion and LoRA imports
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTokenizer, CLIPTextModel, Adafactor
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, toml_path: str):
        config = toml.load(toml_path)

        subset = config['subsets'][0]
        general_args_args = config['general_args']['args']
        dataset_args = config['general_args']['dataset_args']
        optimizer_args = config['optimizer_args']['args']
        saving_args = config['saving_args']['args']
        noise_args = config['noise_args']['args']
        logging_args = config['logging_args']['args']

        # Data settings
        self.data_dir = subset['image_dir']
        self.num_repeats = subset['num_repeats']
        self.image_size = dataset_args['resolution'][0]
        self.batch_size = dataset_args['batch_size']
        self.num_workers = general_args_args['max_data_loader_n_workers']

        # Model settings
        self.model_id = general_args_args['pretrained_model_name_or_path']
        self.pretrained_model_name = "runwayml/stable-diffusion-v1-5"

        # LoRA settings
        self.lora_rank = config['network_args']['args']['network_dim']
        self.lora_alpha = config['network_args']['args']['network_alpha']
        self.lora_dropout = 0.1 # No corresponding field found, default used
        self.target_modules = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

        # Training settings
        self.learning_rate = optimizer_args['learning_rate']
        self.num_epochs = general_args_args['max_train_epochs']
        self.gradient_accumulation_steps = general_args_args['gradient_accumulation_steps']
        self.max_grad_norm = optimizer_args['max_grad_norm']

        # Noise settings
        self.noise_offset = noise_args['noise_offset']

        # Scheduler settings
        self.lr_scheduler = optimizer_args['lr_scheduler']
        # Add scheduler specific parameters here if needed
        # For CosineAnnealingLR, you might need:
        self.lr_scheduler_T_max = self.num_epochs * 1000 # Placeholder, will be updated based on dataloader length
        self.lr_scheduler_eta_min = optimizer_args.get('lr_scheduler_eta_min', 0.0) # For CosineAnnealingLR

        # Output settings
        self.output_dir = saving_args['output_dir']
        self.save_every_n_epochs = saving_args['save_every_n_epochs']
        self.validation_prompts = [
            "Two trucks platooning on highway",
            "Autonomous vehicles in convoy formation",
            "Commercial trucks following in close formation",
            "Platooning vehicles maintaining safe distance"
        ]

        # Logging
        self.log_with = logging_args['log_with']
        self.project_name = logging_args['log_tracker_name']

        # Advanced settings
        self.mixed_precision = general_args_args['mixed_precision']
        self.gradient_checkpointing = general_args_args['gradient_checkpointing']

class EnhancedPlatooningDataset(Dataset):
    """
    Enhanced dataset with support for repeats to simulate larger epochs.
    """

    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split

        self.image_paths = sorted(glob.glob(os.path.join(config.data_dir, "*.png")))
        logger.info(f"Found {len(self.image_paths)} images in {config.data_dir}")

        split_idx = int(1 * len(self.image_paths))
        if split == "train":
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]

        logger.info(f"{self.split} split: {len(self.image_paths)} images")

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        if self.split == "train":
            return len(self.image_paths) * self.config.num_repeats
        return len(self.image_paths)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        image_path = self.image_paths[actual_idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros(3, self.config.image_size, self.config.image_size)

        text_path = image_path.replace(".png", ".txt")
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if '|' in text:
                        text = text.split('|', 1)[1].strip()
            except Exception as e:
                logger.warning(f"Error loading text {text_path}: {e}")
                text = "platooning vehicles on road"
        else:
            text = "platooning vehicles on road"

        return {"pixel_values": image_tensor, "input_ids": text, "image_path": image_path}


class PlatooningLoRATrainer:
    """Trainer adapted to match .toml configuration"""

    def __init__(self, config: Config):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.log_with,
            project_dir=config.output_dir
        )

        os.makedirs(config.output_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(config.project_name)

        self.setup_models()
        self.setup_data()
        self.setup_training()

    def setup_models(self):
        logger.info("Setting up models...")

        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.pretrained_model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.pretrained_model_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.config.pretrained_model_name, subfolder="vae")

        if os.path.exists(self.config.model_id):
            logger.info(f"Loading UNet from local file: {self.config.model_id}")
            pipeline = StableDiffusionPipeline.from_single_file(
                self.config.model_id,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            self.unet = pipeline.unet
        else:
            logger.info(f"Loading UNet from HuggingFace: {self.config.pretrained_model_name}")
            self.unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_name, subfolder="unet")

        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.pretrained_model_name, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

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

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # FIX: Move VAE to device but keep it in fp32 for stability
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)


    def setup_data(self):
        logger.info("Setting up datasets...")
        self.train_dataset = EnhancedPlatooningDataset(self.config, split="train")
        self.val_dataset = EnhancedPlatooningDataset(self.config, split="val")

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=True
        )

    def setup_training(self):
        logger.info("Setting up training components...")
        logger.info("Using Adafactor optimizer")
        self.optimizer = Adafactor(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        # --- MODIFICATION START ---
        # Conditionally create the LR scheduler based on config
        if self.config.lr_scheduler == "constant":
            # For a truly constant learning rate, you don't need a scheduler object
            # or you can use a LambdaLR that returns 1.0 always.
            # However, for simplicity and compatibility with accelerator.prepare,
            # we can create a dummy scheduler or just skip self.lr_scheduler.step()
            # For this case, we'll make a scheduler that just returns 1.0.
            # If you remove this, ensure you also remove self.lr_scheduler.step() in train_step.
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
            logger.info("Using Constant learning rate scheduler.")
        elif self.config.lr_scheduler == "cosine":
            # Assuming T_max is number of total training steps
            # You might need to adjust T_max based on actual steps per epoch
            total_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_training_steps,
                eta_min=self.config.lr_scheduler_eta_min # Use a configurable eta_min
            )
            logger.info(f"Using CosineAnnealingLR with T_max={total_training_steps} and eta_min={self.config.lr_scheduler_eta_min}.")
        # Add more scheduler types here as needed (e.g., "linear", "step")
        else:
            logger.warning(f"Unsupported LR scheduler: {self.config.lr_scheduler}. Defaulting to no scheduler.")
            self.lr_scheduler = None # Or a dummy scheduler that does nothing


        # --- MODIFICATION END ---

        # Prepare scheduler with accelerator only if it's not None
        if self.lr_scheduler:
            self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        else:
            self.unet, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.val_dataloader
            )


    def encode_text(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            texts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.accelerator.device))[0]
        return text_embeddings

    def train_step(self, batch: Dict) -> float:
        with self.accelerator.accumulate(self.unet):
            weight_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

            # FIX: Encode images to latents in fp32 for VAE stability
            with torch.no_grad():
                # Ensure VAE input is float32
                pixel_values_fp32 = batch["pixel_values"].to(dtype=self.vae.dtype)
                latents = self.vae.encode(pixel_values_fp32).latent_dist.sample()

            # Scale and cast latents to the mixed-precision dtype for the UNet
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

            noise = torch.randn_like(latents)
            if self.config.noise_offset > 0:
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            text_embeddings = self.encode_text(batch["input_ids"])
            model_pred = self.unet(noisy_latents, timesteps, text_embeddings, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            # Only step the scheduler if it exists
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.detach().item()

    def validate(self) -> float:
        self.unet.eval()
        total_loss = 0
        num_batches = 0
        weight_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        with torch.no_grad():
            for batch in self.val_dataloader:
                # FIX: Encode images to latents in fp32 for VAE stability
                pixel_values_fp32 = batch["pixel_values"].to(dtype=self.vae.dtype)
                latents = self.vae.encode(pixel_values_fp32).latent_dist.sample()

                # Scale and cast latents to the mixed-precision dtype
                latents = latents * self.vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                text_embeddings = self.encode_text(batch["input_ids"])
                model_pred = self.unet(noisy_latents, timesteps, text_embeddings, return_dict=False)[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_loss += loss.item()
                num_batches += 1

        self.unet.train()
        return total_loss / num_batches if num_batches > 0 else 0

    def save_model(self, epoch_str: str):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch_str}")
        self.accelerator.save_state(save_path)

        if self.accelerator.is_main_process:
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))

            config_dict = {k: v for k, v in vars(self.config).items() if not k.startswith('__') and not callable(v)}
            with open(os.path.join(save_path, "training_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Model saved to {save_path}")

    def train(self):
        logger.info("Starting training...")
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            self.unet.train()
            total_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", disable=not self.accelerator.is_local_main_process)

            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                total_loss += loss
                global_step += 1

                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{total_loss / (step + 1):.4f}",
                    "lr": f"{current_lr:.2e}"
                })

                if self.accelerator.is_main_process:
                    self.accelerator.log({
                        "train_loss": loss,
                        "learning_rate": current_lr,
                    }, step=global_step)

            val_loss = self.validate()
            avg_train_loss = total_loss / len(self.train_dataloader)

            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.accelerator.is_main_process:
                self.accelerator.log({
                    "val_loss": val_loss,
                    "avg_train_loss": avg_train_loss,
                    "epoch": epoch + 1
                }, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best")

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_model(str(epoch + 1))

        self.save_model("final")
        self.accelerator.end_training()
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Enhanced LoRA Training for Platooning Recognition")
    parser.add_argument("--data_dir", type=str, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--num_repeats", type=int, help="Number of times to repeat the dataset per epoch")

    args = parser.parse_args()
    config = Config("Platooning.toml")

    if args.data_dir: config.data_dir = args.data_dir
    if args.output_dir: config.output_dir = args.output_dir
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.lora_rank: config.lora_rank = args.lora_rank
    if args.num_repeats: config.num_repeats = args.num_repeats


    trainer = PlatooningLoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()