import argparse #To enable running the script from the command line
from Experiments import run_experiment
import torch
from Downscaling_Dataset_Prep import DownscalingDataset
from Downscaling_Dataset_Prep import PairedDataset
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from Train_Test_Val import split_by_decade
from config import CONFIG #configuration file with the paths and variable names

def main():
    paths = CONFIG["input_paths"]
    var_names = CONFIG["var_names"]

    # Datasets
    precip_input = xr.open_dataset(paths["precip_input"])
    temp_input = xr.open_dataset(paths["temp_input"])
    precip_target = xr.open_dataset(paths["precip_target"])
    temp_target = xr.open_dataset(paths["temp_target"])

    # Image-target pairs creation
    torch_precip_dataset = DownscalingDataset(precip_input, precip_target, var_names["precip_input"], var_names["precip_target"])
    torch_temp_dataset = DownscalingDataset(temp_input, temp_target, var_names["temp_input"], var_names["temp_target"])

    # Split indices by train-test -val scheme : every decade 70% training 20% val and 10 percent testing

    times = precip_input['time'].values
    train_idx, val_idx, test_idx = split_by_decade(times)

    # Subset datasets into training  testing and validation sets based on the schema
    precip_train = Subset(torch_precip_dataset, train_idx)
    temp_train = Subset(torch_temp_dataset, train_idx)
    precip_val = Subset(torch_precip_dataset, val_idx)
    temp_val = Subset(torch_temp_dataset, val_idx)
    precip_test = Subset(torch_precip_dataset, test_idx)
    temp_test = Subset(torch_temp_dataset, test_idx)

    # Combine into paired datasets
    train_dataset = PairedDataset(precip_train, temp_train)
    val_dataset = PairedDataset(precip_val, temp_val)
    test_dataset = PairedDataset(precip_test, temp_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

if __name__ == "__main__":
    main()
