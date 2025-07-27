# Platooning-Lora Training Pipeline
# Simplified Makefile with essential commands

# Configuration
VENV = .venv
PYTHON = $(VENV)/bin/python
OUTPUT_DIR = platooning_lora_v2
DATASET_DIR = "Platooning images-resized"
BASE_MODEL = v1-5-pruned.safetensors

# Default parameters
EPOCHS ?= 50
BATCH_SIZE ?= 1
LEARNING_RATE ?= 1e-4
LORA_RANK ?= 16

# PHONY targets
.PHONY: init train-quick train clean install update-deps add-dep info

# Default target - show help
.DEFAULT_GOAL := help

help:
	@echo "Platooning LoRA Training Pipeline"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make init         - Initialize project and install dependencies"
	@echo "  make install      - Install/sync dependencies (alias for init)"
	@echo "  make update-deps  - Update dependencies to latest versions"
	@echo "  make info         - Show project and environment information"
	@echo "  make train-quick  - Quick training with 10 epochs for testing"
	@echo "  make train        - Full training with default parameters (50 epochs)"
	@echo "  make clean        - Clean all generated files and environment"
	@echo ""
	@echo "Parameters (can be overridden):"
	@echo "  EPOCHS=$(EPOCHS)           - Number of training epochs"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)         - Training batch size"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)    - Learning rate"
	@echo "  LORA_RANK=$(LORA_RANK)          - LoRA rank parameter"
	@echo ""
	@echo "Examples:"
	@echo "  make train EPOCHS=20 LEARNING_RATE=1e-5"
	@echo "  make train-quick BATCH_SIZE=2"
	@echo "  make add-dep PACKAGE=wandb    # Add new dependency"

# Initialize project
init:
	@echo "🚀 Initializing Platooning-Lora project..."
	@echo "📦 Installing dependencies from uv.lock and pyproject.toml..."
	uv sync --no-install-project
	@echo "🔍 Checking setup..."
	@test -f .venv/bin/python && echo "✅ Virtual environment created" || echo "❌ Virtual environment creation failed"
	@.venv/bin/python -c "import torch, transformers, diffusers, peft" 2>/dev/null && echo "✅ Core dependencies installed" || echo "❌ Core dependencies missing"
	@test -f $(BASE_MODEL) && echo "✅ Base model found ($(shell du -h $(BASE_MODEL) | cut -f1))" || echo "⚠️  Base model not found - will download from HuggingFace"
	@test -d $(DATASET_DIR) && echo "✅ Dataset directory found ($(shell ls $(DATASET_DIR)/*.png 2>/dev/null | wc -l) images)" || echo "❌ Dataset directory not found"
	@echo "✅ Project initialized successfully!"
	@echo "💡 Virtual environment created at .venv/"

# Quick training for testing (10 epochs)
train-quick:
	@test -f $(VENV)/bin/activate || (echo "❌ Environment not initialized. Run 'make init' first." && exit 1)
	@echo "⚡ Starting quick training (10 epochs)..."
	uv run python run_training.py --epochs 10 --batch_size $(BATCH_SIZE) --output_dir $(OUTPUT_DIR)
	@echo "✅ Quick training completed! Check $(OUTPUT_DIR)/ for results"

# Full training with configurable parameters
train:
	@test -f $(VENV)/bin/activate || (echo "❌ Environment not initialized. Run 'make init' first." && exit 1)
	@echo "🚀 Starting full training with $(EPOCHS) epochs..."
	uv run python run_training.py \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--learning_rate $(LEARNING_RATE) \
		--lora_rank $(LORA_RANK) \
		--output_dir $(OUTPUT_DIR)
	@echo "✅ Training completed! Check $(OUTPUT_DIR)/ for results"

# Clean up all generated files and environment
clean:
	@echo "🧹 Cleaning up project..."
	@echo "This will remove the virtual environment, training outputs, and generated files."
	@printf "Continue? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]|[Yy][Ee][Ss]) \
			rm -rf $(VENV) $(OUTPUT_DIR) __pycache__ *.png evaluation_results/ uv.lock; \
			echo "✅ Cleanup completed" ;; \
		*) \
			echo "❌ Cleanup cancelled" ;; \
	esac

# Alias for init
install: init

# Update dependencies to latest versions
update-deps:
	@echo "📦 Updating dependencies to latest versions..."
	uv sync --upgrade --no-install-project
	@echo "✅ Dependencies updated!"

# Add a new dependency
add-dep:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "❌ Please specify a package: make add-dep PACKAGE=package_name"; \
		exit 1; \
	fi
	@echo "➕ Adding $(PACKAGE)..."
	uv add $(PACKAGE)
	@echo "✅ $(PACKAGE) added and lock file updated!"

# Show project information
info:
	@echo "📋 Platooning-LoRA Project Information:"
	@echo "   Project: platooning-lora"
	@echo "   Python: $(shell python --version 2>/dev/null || echo 'Not activated')"
	@echo "   UV: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "   Virtual Environment: $(shell echo $$VIRTUAL_ENV || echo 'Not activated (.venv available)')"
	@echo "   Dependencies: $(shell [ -f uv.lock ] && echo 'Locked ($(shell grep -c '\[\[package\]\]' uv.lock) packages)' || echo 'Not locked')"
	@echo "   Base Model: $(shell [ -f $(BASE_MODEL) ] && echo 'Present ($(shell du -h $(BASE_MODEL) | cut -f1))' || echo 'Missing')"
	@echo "   Dataset: $(shell [ -d $(DATASET_DIR) ] && echo 'Present ($(shell ls $(DATASET_DIR)/*.png 2>/dev/null | wc -l) images)' || echo 'Missing')"
	@echo "   Training Output: $(shell [ -d $(OUTPUT_DIR) ] && echo 'Directory exists' || echo 'Not created yet')"
