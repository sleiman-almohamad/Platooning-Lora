#!/usr/bin/env python3
"""
Enhanced Inference Script for Platooning LoRA Model
Supports multiple generation modes and evaluation metrics
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image
import argparse
import os
import json
from typing import List, Optional, Dict
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlatooningInference:
    """Enhanced inference class with multiple generation modes"""
    
    def __init__(
        self, 
        base_model_path: str = "v1-5-pruned.safetensors",
        lora_path: str = "platooning_lora_v2/best",
        device: str = "auto"
    ):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the base model and LoRA adapters"""
        logger.info("Loading base Stable Diffusion model...")
        
        try:
            # Load base pipeline
            if os.path.exists(self.base_model_path):
                self.pipeline = StableDiffusionPipeline.from_single_file(
                    self.base_model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                logger.info("Local model not found, loading from HuggingFace...")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LoRA adapters if available
            if os.path.exists(self.lora_path):
                logger.info(f"Loading LoRA adapters from {self.lora_path}")
                self.pipeline.unet = PeftModel.from_pretrained(
                    self.pipeline.unet, 
                    self.lora_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                logger.info("LoRA adapters loaded successfully")
            else:
                logger.warning(f"LoRA path {self.lora_path} not found, using base model only")
            
            # Optimize pipeline
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable memory efficient attention if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled")
            except:
                logger.info("xformers not available, using default attention")
            
            # Enable CPU offload for low VRAM
            if self.device.type == "cuda":
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Model CPU offload enabled")
                except:
                    pass
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_single(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a single image from prompt"""
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.autocast(self.device.type):
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            ).images[0]
        
        return image
    
    def generate_batch(
        self,
        prompts: List[str],
        output_dir: str = "generated_images",
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple images from a list of prompts"""
        
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                image = self.generate_single(prompt, **kwargs)
                images.append(image)
                
                # Save image
                filename = f"generated_{i+1:03d}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                # Save prompt
                prompt_file = os.path.join(output_dir, f"generated_{i+1:03d}.txt")
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
                
                logger.info(f"Saved: {filepath}")
                
            except Exception as e:
                logger.error(f"Error generating image for prompt '{prompt}': {e}")
                continue
        
        return images
    
    def evaluate_on_dataset(
        self,
        dataset_dir: str,
        output_dir: str = "evaluation_results",
        num_samples: Optional[int] = None
    ) -> Dict:
        """Evaluate model on dataset by regenerating images from text descriptions"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all text files in dataset
        text_files = list(Path(dataset_dir).glob("*.txt"))
        if num_samples:
            text_files = text_files[:num_samples]
        
        logger.info(f"Evaluating on {len(text_files)} samples")
        
        results = {
            "total_samples": len(text_files),
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0
        }
        
        total_time = 0
        
        for i, text_file in enumerate(text_files):
            try:
                # Load original text
                with open(text_file, 'r') as f:
                    prompt = f.read().strip()
                    if '|' in prompt:
                        prompt = prompt.split('|', 1)[1].strip()
                
                # Generate image
                start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
                
                if start_time:
                    start_time.record()
                
                image = self.generate_single(prompt)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                    total_time += generation_time
                
                # Save generated image
                output_path = os.path.join(output_dir, f"eval_{i+1:03d}_generated.png")
                image.save(output_path)
                
                # Save prompt
                prompt_path = os.path.join(output_dir, f"eval_{i+1:03d}_prompt.txt")
                with open(prompt_path, 'w') as f:
                    f.write(prompt)
                
                # Copy original image for comparison
                original_image_path = text_file.with_suffix('.png')
                if original_image_path.exists():
                    original_copy_path = os.path.join(output_dir, f"eval_{i+1:03d}_original.png")
                    Image.open(original_image_path).save(original_copy_path)
                
                results["successful_generations"] += 1
                logger.info(f"Generated {i+1}/{len(text_files)}: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate image for {text_file}: {e}")
                results["failed_generations"] += 1
        
        if results["successful_generations"] > 0:
            results["average_generation_time"] = total_time / results["successful_generations"]
        
        # Save evaluation results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Enhanced Platooning LoRA Inference")
    parser.add_argument("--mode", choices=["single", "batch", "evaluate"], default="single",
                       help="Inference mode")
    parser.add_argument("--prompt", type=str, 
                       default="Two trucks platooning on highway in formation",
                       help="Prompt for single image generation")
    parser.add_argument("--prompts_file", type=str,
                       help="File containing prompts for batch generation")
    parser.add_argument("--dataset_dir", type=str, default="./Platooning images-resized",
                       help="Dataset directory for evaluation")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                       help="Output directory")
    parser.add_argument("--base_model", type=str, default="v1-5-pruned.safetensors",
                       help="Base model path")
    parser.add_argument("--lora_path", type=str, default="platooning_lora_v2/best",
                       help="LoRA model path")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_samples", type=int, help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = PlatooningInference(
        base_model_path=args.base_model,
        lora_path=args.lora_path
    )
    
    if args.mode == "single":
        logger.info(f"Generating single image for prompt: {args.prompt}")
        image = inferencer.generate_single(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "generated_image.png")
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
    
    elif args.mode == "batch":
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default platooning prompts
            prompts = [
                "Two trucks platooning on highway in tight formation",
                "Autonomous vehicles convoy maintaining safe following distance",
                "Commercial trucks in platooning formation on interstate",
                "Fleet of trucks traveling in coordinated convoy",
                "Platooning vehicles demonstrating automated following behavior"
            ]
        
        logger.info(f"Generating {len(prompts)} images")
        inferencer.generate_batch(
            prompts=prompts,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
    
    elif args.mode == "evaluate":
        logger.info(f"Evaluating model on dataset: {args.dataset_dir}")
        results = inferencer.evaluate_on_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
        
        print("\n=== Evaluation Results ===")
        print(f"Total samples: {results['total_samples']}")
        print(f"Successful generations: {results['successful_generations']}")
        print(f"Failed generations: {results['failed_generations']}")
        print(f"Success rate: {results['successful_generations']/results['total_samples']*100:.1f}%")
        if results['average_generation_time'] > 0:
            print(f"Average generation time: {results['average_generation_time']:.2f}s")


if __name__ == "__main__":
    main()
