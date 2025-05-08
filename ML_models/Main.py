import argparse
import torch
from torch.utils.data import DataLoader
import xarray as xr
import os
import wandb

from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset, PairedDataset
from config import CONFIG

def load_dataset(input_path, target_path, input_var, target_var):
    input_ds = xr.open_dataset(input_path)
    target_ds = xr.open_dataset(target_path)
    return DownscalingDataset(input_ds, target_ds, input_var, target_var)

def main():

    paths = CONFIG["input_paths"]
    var_names = CONFIG["var_names"]
    quick_test = CONFIG.get("quick_test", False)

    wandb.init(
        project="Deterministic UNet",
        name="Experiment_Quick_Test" if quick_test else "Experiment_Full_Run",
        config=CONFIG
    )

    precip_train = load_dataset(paths["precip_train_input"], paths["precip_train_target"], var_names["precip_input"], var_names["precip_target"])
    temp_train = load_dataset(paths["temp_train_input"], paths["temp_train_target"], var_names["temp_input"], var_names["temp_target"])

    precip_val = load_dataset(paths["precip_val_input"], paths["precip_val_target"], var_names["precip_input"], var_names["precip_target"])
    temp_val = load_dataset(paths["temp_val_input"], paths["temp_val_target"], var_names["temp_input"], var_names["temp_target"])

    precip_test = load_dataset(paths["precip_test_input"], paths["precip_test_target"], var_names["precip_input"], var_names["precip_target"])
    temp_test = load_dataset(paths["temp_test_input"], paths["temp_test_target"], var_names["temp_input"], var_names["temp_target"])

    train_dataset = PairedDataset(precip_train, temp_train)
    val_dataset = PairedDataset(precip_val, temp_val)
    test_dataset = PairedDataset(precip_test, temp_test)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history, final_val_loss = run_experiment(train_dataset, val_dataset, quick_test=quick_test, num_epochs=30)

    # Final validation loss
    wandb.log({"final_validation_loss": final_val_loss})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model for downscaling.")
    parser.add_argument("--quick_test", action="store_true", help="Override config to run a quick test")
    args = parser.parse_args()

    # If CLI --quick_test is given, override CONFIG
    if args.quick_test:
        CONFIG["quick_test"] = True

    main()
