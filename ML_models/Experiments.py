import sys
sys.path.append("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/")
from UNet import UNet
import torch
import torch.nn as nn
from train import train_model

def run_experiment(train_dataset, val_dataset, quick_test=True):
    model = UNet(in_channels=2, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if quick_test:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, range(100)),
            batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_dataset, range(50)),
            batch_size=32, shuffle=False
        )
        num_epochs = 10
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        num_epochs = 50

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        quick_test=quick_test
    )

    return trained_model, history
