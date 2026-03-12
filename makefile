# Makefile for Platooning LoRA Project
# Cross-platform version with: init, train, clean

# --- OS Detection ---
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	RM := del /Q
	RMDIR := rmdir /S /Q
	MKDIR := mkdir
	TOUCH := type nul >
	SHELL := cmd
	EXE_EXT := .exe
	PATH_SEP := \\
	NULL_DEV := nul
else
	DETECTED_OS := $(shell uname -s)
	RM := rm -f
	RMDIR := rm -rf
	MKDIR := mkdir -p
	TOUCH := touch
	SHELL := /bin/bash
	EXE_EXT :=
	PATH_SEP := /
	NULL_DEV := /dev/null
endif

# --- Configuration ---
VENV_DIR := platooning_lora.egg-info
ifeq ($(DETECTED_OS),Windows)
	UV_EXE := $(shell where uv 2>$(NULL_DEV) || echo "uv$(EXE_EXT)")
else
	UV_EXE := $(shell command -v uv || echo "$(HOME)/.cargo/bin/uv")
endif
BASE_MODEL := $(firstword $(wildcard *.safetensors))
BASE_MODEL_URL := https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors
DATASET_DIR := dataset
OUTPUT_DIR := Output_Lora_training
#CONFIG_FILE := training_pars_lora.toml

.PHONY: init train clean show-os

# --- Main Targets ---
init: show-os | install-uv create-venv check-dataset download-model
	@echo "Initialization complete"

show-os:
	@echo "--- Detected OS: $(DETECTED_OS) ---"

#train: $(VENV_DIR)/touchfile
#	@echo "--- Starting training ---"
#ifeq ($(DETECTED_OS),Windows)
#	@powershell -Command "(Get-Content $(CONFIG_FILE)) -replace '^pretrained_model_name_or_path = .*', 'pretrained_model_name_or_path = \"$(BASE_MODEL)\"' | Set-Content $(CONFIG_FILE)"
#else
#	@sed -i "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$(BASE_MODEL)\"|" $(CONFIG_FILE)
#endif
#	@$(UV_EXE) run train_network.py --config_file=$(CONFIG_FILE)
#	@echo "Training completed"

train_lora:
	@echo "--- Starting training ---"
ifeq ($(DETECTED_OS),Windows)
	@powershell -Command "(Get-Content $(training_pars_lora.toml)) -replace '^pretrained_model_name_or_path = .*', 'pretrained_model_name_or_path = \"$(BASE_MODEL)\"| Set-Content $(CONFIG_FILE)"
else
	@sed -i "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$(BASE_MODEL)\"|" $(training_pars_lora.toml)
endif
	@$(UV_EXE) run train_network.py --config_file=$(training_pars_lora.toml)
	@echo "Training completed"

train_dora:
	@echo "--- Starting training ---"
ifeq ($(DETECTED_OS),Windows)
	@powershell -Command "(Get-Content $(training_pars_dora.toml)) -replace '^pretrained_model_name_or_path = .*', 'pretrained_model_name_or_path = \"$(BASE_MODEL)\"| Set-Content $(CONFIG_FILE)"
else
	@sed -i "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$(BASE_MODEL)\"|" $(training_pars_dora.toml)
endif
	@$(UV_EXE) run train_network.py --config_file=$(training_pars_dora.toml)
	@echo "Training completed"

clean:
	@echo "--- Cleaning project ---"
ifeq ($(DETECTED_OS),Windows)
	@if exist "$(VENV_DIR)" $(RMDIR) "$(VENV_DIR)"
	@if exist "$(OUTPUT_DIR)" $(RMDIR) "$(OUTPUT_DIR)"
else
	@$(RMDIR) $(VENV_DIR) $(OUTPUT_DIR) 2>$(NULL_DEV) || true
endif
	@echo "Cleanup complete"

# --- Support Targets ---
install-uv:
ifeq ($(DETECTED_OS),Windows)
	@where uv >$(NULL_DEV) 2>&1 || ( \
		echo Installing uv for Windows... && \
		powershell -Command "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression" \
	)
else
	@command -v $(UV_EXE) >$(NULL_DEV) 2>&1 || ( \
		echo "Installing uv for $(DETECTED_OS)..." && \
		curl -LsSf https://astral.sh/uv/install.sh | sh \
	)
endif

create-venv: $(VENV_DIR)/touchfile

$(VENV_DIR)/touchfile: pyproject.toml
	@echo "--- Creating virtual environment ---"
	@$(UV_EXE) venv $(VENV_DIR)
	@$(UV_EXE) pip install -e .
ifeq ($(DETECTED_OS),Windows)
	@$(TOUCH) "$(VENV_DIR)\touchfile"
else
	@$(TOUCH) $(VENV_DIR)/touchfile
endif
	@echo "Virtual environment ready"

check-dataset:
	@echo "--- Checking dataset ---"
ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(DATASET_DIR)" ( \
		echo Error: Dataset directory '$(DATASET_DIR)' not found && \
		exit /b 1 \
	)
else
	@if [ ! -d "$(DATASET_DIR)" ]; then \
		echo "Error: Dataset directory '$(DATASET_DIR)' not found"; \
		exit 1; \
	fi
endif
	@echo "Dataset verified"

download-model:
	@echo "--- Checking base model ---"
ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(BASE_MODEL)" ( \
		echo Downloading base model... && \
		powershell -Command "Invoke-WebRequest -Uri '$(BASE_MODEL_URL)' -OutFile '$(BASE_MODEL)' -UseBasicParsing" || ( \
			echo Error downloading model && exit /b 1 \
		) \
	)
else
	@if [ ! -f "$(BASE_MODEL)" ]; then \
		echo "Downloading base model..."; \
		(command -v wget >$(NULL_DEV) 2>&1 && wget -q --show-progress -O $(BASE_MODEL) $(BASE_MODEL_URL)) || \
		(command -v curl >$(NULL_DEV) 2>&1 && curl -L -o $(BASE_MODEL) $(BASE_MODEL_URL)) || \
		(echo "Error: Neither wget nor curl found for downloading model"; exit 1); \
	fi
endif
	@echo "Base model ready"
