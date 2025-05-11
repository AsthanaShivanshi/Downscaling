import torch
from torch.utils.data import Dataset
import numpy as np

class DownscalingDataset(Dataset):
    def __init__(self, input_ds, target_ds, config, data_type):
        """
        data_type: one of ['precip', 'temp']
        config: full merged config dictionary with paths from the untracked file and variables from the tracked config.yaml file
        """
        var_inputs = config["variables"]["input"][data_type]
        var_targets = config["variables"]["target"][data_type]

        self.input = input_ds[var_inputs]
        self.target = target_ds[var_targets]

        self.handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
        self.nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, index):
        input_img = self.input.isel(time=index).values
        target_img = self.target.isel(time=index).values

        if self.handle_nan:
            input_img = np.nan_to_num(input_img, nan=self.nan_value)
            target_img = np.nan_to_num(target_img, nan=self.nan_value)

        input_img = torch.tensor(input_img).unsqueeze(0).float()
        target_img = torch.tensor(target_img).unsqueeze(0).float()

        return input_img, target_img


class PairedDataset(Dataset):
    def __init__(self, precip_dataset, temp_dataset):
        assert len(precip_dataset) == len(temp_dataset), "Datasets must be the same length."
        self.precip_dataset = precip_dataset
        self.temp_dataset = temp_dataset

    def __len__(self):
        return len(self.precip_dataset)

    def __getitem__(self, idx):
        precip_input, precip_target = self.precip_dataset[idx]
        temp_input, temp_target = self.temp_dataset[idx]

        input_combined = torch.cat([precip_input, temp_input], dim=0)
        target_combined = torch.cat([precip_target, temp_target], dim=0)

        return input_combined, target_combined
