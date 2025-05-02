import argparse  # To enable running the script from the command line
import torch
from torch.utils.data import DataLoader
import xarray as xr
import os

from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset, PairedDataset
from config import CONFIG  # Configuration file with the paths and variable names

def load_dataset(input_path, target_path, input_var, target_var):
    input_ds = xr.open_dataset(input_path)
    target_ds = xr.open_dataset(target_path)
    return DownscalingDataset(input_ds, target_ds, input_var, target_var)

def main(quick_test=True):
    paths = CONFIG["input_paths"]
    var_names = CONFIG["var_names"]

    # Loading split datasets
    precip_train = load_dataset(paths["precip_train_input"], paths["precip_train_target"], var_names["precip_input"], var_names["precip_target"])
    temp_train = load_dataset(paths["temp_train_input"], paths["temp_train_target"], var_names["temp_input"], var_names["temp_target"])

    precip_val = load_dataset(paths["precip_val_input"], paths["precip_val_target"], var_names["precip_input"], var_names["precip_target"])
    temp_val = load_dataset(paths["temp_val_input"], paths["temp_val_target"], var_names["temp_input"], var_names["temp_target"])

    precip_test = load_dataset(paths["precip_test_input"], paths["precip_test_target"], var_names["precip_input"], var_names["precip_target"])
    temp_test = load_dataset(paths["temp_test_input"], paths["temp_test_target"], var_names["temp_input"], var_names["temp_target"])

    # Making paired combinations of precip and temp
    train_dataset = PairedDataset(precip_train, temp_train)
    val_dataset = PairedDataset(precip_val, temp_val)
    test_dataset = PairedDataset(precip_test, temp_test)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Run training
    model, history, final_val_loss = run_experiment(train_dataset, val_dataset, quick_test=quick_test, num_epochs=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model for downscaling.")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test with limited data.")
    args = parser.parse_args()

    main(quick_test=args.quick_test)
