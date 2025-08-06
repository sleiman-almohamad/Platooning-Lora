# Makefile for Platooning LoRA Project
# Simple version with: init, train, clean

# --- Configuration ---
VENV_DIR := platooning_lora.egg-info
UV_EXE := $(shell command -v uv || echo "$(HOME)/.cargo/bin/uv")
BASE_MODEL := $(firstword $(wildcard *.safetensors))
BASE_MODEL_URL := https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors
DATASET_DIR := dataset
OUTPUT_DIR := Output_Lora_training
CONFIG_FILE := training_pars.toml

.PHONY: init train clean

# --- Main Targets ---
init: | install-uv create-venv check-dataset download-model
	@echo "✓ Initialization complete"

train: $(VENV_DIR)/touchfile
	@echo "--- Starting training ---"
	@sed -i "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$(BASE_MODEL)\"|" $(CONFIG_FILE)
	@$(UV_EXE) run train_network.py --config_file=$(CONFIG_FILE)
	@echo "✓ Training completed"

clean:
	@echo "--- Cleaning project ---"
	@rm -rf $(VENV_DIR) $(OUTPUT_DIR)
	@echo "✓ Cleanup complete"

# --- Support Targets ---
install-uv:
	@if ! command -v $(UV_EXE) >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

create-venv: $(VENV_DIR)/touchfile

$(VENV_DIR)/touchfile: pyproject.toml
	@echo "--- Creating virtual environment ---"
	@$(UV_EXE) venv $(VENV_DIR)
	@$(UV_EXE) pip install -e .
	@touch $(VENV_DIR)/touchfile
	@echo "✓ Virtual environment ready"

check-dataset:
	@echo "--- Checking dataset ---"
	@if [ ! -d "$(DATASET_DIR)" ]; then \
		echo "Error: Dataset directory '$(DATASET_DIR)' not found"; \
		exit 1; \
	fi
	@echo "✓ Dataset verified"

download-model:
	@echo "--- Checking base model ---"
	@if [ ! -f "$(BASE_MODEL)" ]; then \
		echo "Downloading base model..."; \
		wget -q --show-progress -O $(BASE_MODEL) $(BASE_MODEL_URL) || \
		(echo "Error downloading model"; exit 1); \
	fi
	@echo "✓ Base model ready"
