import os
import shutil
import subprocess
import toml
from pathlib import Path

# Grid search parameters
DATASET_SIZES = [5, 10, 15, 25, 50]
RANKS = [4, 8, 16, 32]
MODELS = ["lora", "dora"]

BASE_DIR = Path("/home/slimanutd/sleiman/Master-Thesis/Platooning-Lora_vs_Dora")
DATASET_BASE = BASE_DIR / "dataset" / "1_platooning"
OUTPUT_BASE = BASE_DIR / "grid_search_outputs"
TEMP_DATASET = BASE_DIR / "gs_temp_dataset"

def setup_temp_dataset(size):
    """Sets up a temporary dataset directory in the format expected by the training script."""
    if TEMP_DATASET.exists():
        shutil.rmtree(TEMP_DATASET)
    
    # We use 1 repeat as the base, the 'size' is handled by picking the right source folder
    img_dir = TEMP_DATASET / "1_platooning"
    img_dir.mkdir(parents=True)
    
    src_dir = DATASET_BASE / f"dataset_{size}"
    if not src_dir.exists():
        print(f"Warning: Source dataset directory {src_dir} not found!")
        return False
    
    for item in src_dir.iterdir():
        if item.is_file():
            # Use symlinks to save space and time
            os.symlink(item, img_dir / item.name)
    return True

def run_training(model_type, rank, size):
    """Generates a config and runs the training for a specific combination."""
    run_id = f"{model_type}_r{rank}_data{size}"
    print(f"\n>>> Starting Run: {run_id}")
    
    # Select base config
    base_config_path = BASE_DIR / f"training_pars_{model_type}.toml"
    if not base_config_path.exists():
        print(f"Error: Base config {base_config_path} not found!")
        return
    
    config = toml.load(base_config_path)
    
    # Update parameters for this run
    config['network']['network_dim'] = rank
    config['network']['network_alpha'] = rank / 2  # Often alpha is set to dim/2 or dim
    config['general']['train_data_dir'] = str(TEMP_DATASET)
    
    run_output_dir = OUTPUT_BASE / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    config['save']['output_dir'] = str(run_output_dir)
    config['save']['output_name'] = run_id
    config['logging']['logging_dir'] = str(run_output_dir / "Logs")
    config['logging']['log_tracker_name'] = run_id
    
    # Save temporary config
    temp_config_file = BASE_DIR / f"temp_config_{run_id}.toml"
    with open(temp_config_file, "w") as f:
        toml.dump(config, f)
    
    # Execute training
    try:
        cmd = ["uv", "run", "python", "-m", "accelerate.commands.launch", "train_network.py", f"--config_file={temp_config_file}"]
        subprocess.run(cmd, check=True)
        print(f"<<< Finished Run: {run_id}")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error in Run {run_id}: {e}")
    finally:
        if temp_config_file.exists():
            temp_config_file.unlink()

def main():
    OUTPUT_BASE.mkdir(exist_ok=True)
    
    for model in MODELS:
        for size in DATASET_SIZES:
            # Set up the dataset once for this size
            if not setup_temp_dataset(size):
                continue
            
            for rank in RANKS:
                run_training(model, rank, size)
    
    # Final cleanup
    if TEMP_DATASET.exists():
        shutil.rmtree(TEMP_DATASET)
    print("\nGrid Search Complete!")

if __name__ == "__main__":
    main()
