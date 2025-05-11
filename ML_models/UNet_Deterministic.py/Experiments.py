import sys
sys.path.append("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/Cyclical_LR_UNet")
from UNet import UNet
import torch
import torch.nn as nn
from Train import train_model, checkpoint_save
import os
import wandb
from torch.optim.lr_scheduler import CyclicLR

def run_experiment(train_dataset, val_dataset, quick_test=False, num_epochs=30):
    # Initialize W&B
    wandb.init(project="unet_downscaling", name="CLR_experiment", config={
        "optimizer": "Adam",
        "loss": "MSE",
        "base_lr": 1e-4,
        "max_lr": 1e-3,
        "scheduler": "CyclicLR",
        "mode": "triangular",
        "epochs": num_epochs
    })

    model = UNet(in_channels=2, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=1000, mode="triangular")
    criterion = nn.MSELoss()

    wandb.watch(model, log="all", log_freq=100)

    if quick_test:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, range(100)),
            batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_dataset, range(30)),
            batch_size=32, shuffle=False
        )
        num_epochs = 20
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=scheduler,
        num_epochs=num_epochs,
        quick_test=quick_test
    )

    final_val_loss = history['val_loss'][-1]
    checkpoint_save(
        model, optimizer, epoch=num_epochs, loss=final_val_loss,
        path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/checkpoints/CyclicalLR_100samples__model_UNet_01.pth"
    )

    return trained_model, history, final_val_loss
