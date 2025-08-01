# Platooning-Lora Training Pipeline
# Simplified Makefile with essential commands

# Configuration - Parameters are now managed within the script
VENV = .venv
PYTHON = $(VENV)/bin/python
BASE_MODEL = v1-5-pruned.safetensors

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
	@echo "  make train        - Full training using Platooning.toml config"
	@echo "  make clean        - Clean all generated files and environment"
	@echo ""
	@echo ""
	@echo "Examples:"
	@echo "  make train                               # Run training with TOML config"
	@echo "  make train-quick                         # Quick test training"
	@echo "  make add-dep PACKAGE=wandb               # Add new dependency"
	@echo ""
	@echo "Note: All training parameters are now managed in Platooning.toml."
	@echo "      Edit the TOML file to modify training configuration."

# Initialize project
init:
	@echo "🚀 Initializing Platooning-Lora project..."
	@echo "📦 Installing dependencies from uv.lock and pyproject.toml..."
	uv sync --no-install-project
	@echo "📦 Adding toml dependency for configuration parsing..."
	uv add toml
	@echo "🔍 Checking setup..."
	@test -f .venv/bin/python && echo "✅ Virtual environment created" || echo "❌ Virtual environment creation failed"
	@.venv/bin/python -c "import torch, transformers, diffusers, peft, toml" 2>/dev/null && echo "✅ Core dependencies installed" || echo "❌ Core dependencies missing"
	@test -f $(BASE_MODEL) && echo "✅ Base model found ($(shell du -h $(BASE_MODEL) | cut -f1))" || echo "⚠️  Base model not found - will download from HuggingFace"
	@echo "ℹ️  Dataset directory will be read from Platooning.toml configuration"
	@test -f Platooning.toml && echo "✅ Configuration file found" || echo "❌ Platooning.toml not found"
	@echo "✅ Project initialized successfully!"
	@echo "💡 Virtual environment created at .venv/"

# Quick training for testing (10 epochs)
train-quick:
	@test -f $(VENV)/bin/activate || (echo "❌ Environment not initialized. Run 'make init' first."  exit 1)
	@test -f Platooning.toml || (echo "❌ Platooning.toml not found. Please ensure configuration file exists."  exit 1)
	@echo "⚡ Starting quick training (10 epochs)..."
	uv run python train_new.py
	@echo "✅ Quick training completed! Check Platooning.toml configuration for results"

# Full training using script parameters
train:
	@test -f $(VENV)/bin/activate || (echo "❌ Environment not initialized. Run 'make init' first."  exit 1)
	@test -f Platooning.toml || (echo "❌ Platooning.toml not found. Please ensure configuration file exists."  exit 1)
	@echo "🚀 Starting full training using script parameters..."
	uv run python train_new.py
	@echo "✅ Training completed using script parameters! Check Platooning.toml configuration for results"

# Clean up all generated files and environment
clean:
	@echo "🧹 Cleaning up project..."
	@echo "This will remove the virtual environment, training outputs, and generated files."
	@printf "Continue? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]|[Yy][Ee][Ss]) \
			rm -rf $(VENV) __pycache__ *.png evaluation_results/ uv.lock; \
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
	@echo "   Dataset: Check Platooning.toml for dataset configuration"
	@echo "   Configuration: $(shell [ -f Platooning.toml ] && echo 'Platooning.toml found' || echo 'Platooning.toml missing')"
	@echo "   Training Output: Check Platooning.toml for output directory configuration"
