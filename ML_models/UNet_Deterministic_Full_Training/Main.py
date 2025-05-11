import argparse
import torch
from torch.utils.data import DataLoader
import xarray as xr
import wandb
from pathlib import Path

from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset, PairedDataset
from config_loader import load_config

def load_dataset(input_path, target_path, config, data_type):
    input_ds = xr.open_dataset(input_path)
    target_ds = xr.open_dataset(target_path)
    return DownscalingDataset(input_ds, target_ds, config, data_type)

def main(config):
    paths = config["data"]
    var_names = config["variables"]
    quick_test = config["experiment"].get("quick_test", False)

    root = Path(paths["root"])
    precip_train = load_dataset(root / paths["train"]["inputs"]["precip"], root / paths["train"]["targets"]["precip"], config, "precip")
    temp_train = load_dataset(root / paths["train"]["inputs"]["temp"], root / paths["train"]["targets"]["temp"], config, "temp")

    precip_val = load_dataset(root / paths["val"]["inputs"]["precip"], root / paths["val"]["targets"]["precip"], config, "precip")
    temp_val = load_dataset(root / paths["val"]["inputs"]["temp"], root / paths["val"]["targets"]["temp"], config, "temp")

    precip_test = load_dataset(root / paths["test"]["inputs"]["precip"], root / paths["test"]["targets"]["precip"], config, "precip")
    temp_test = load_dataset(root / paths["test"]["inputs"]["temp"], root / paths["test"]["targets"]["temp"], config, "temp")

    # Combine datasets
    train_dataset = PairedDataset(precip_train, temp_train)
    val_dataset = PairedDataset(precip_val, temp_val)
    test_dataset = PairedDataset(precip_test, temp_test)
    
    print(f"Using learning rate scheduler: {config['train'].get('scheduler', 'CyclicLR')}")

    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history, final_val_loss = run_experiment(train_dataset, val_dataset, config=config)

    # Final logging
    wandb.log({"final_validation_loss": final_val_loss})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model for downscaling.")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test (overrides config)")
    parser.add_argument("--config", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/UNet_Deterministic_Full_Training/config.yaml", help="Path to experiment config file")
    parser.add_argument("--paths", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/UNet_Deterministic_Full_Training/.paths.yaml", help="Path to private file paths config")
    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config, args.paths)

    # Override quick_test flag if specified via CLI
    if args.quick_test:
        config["experiment"]["quick_test"] = True

    main(config)
