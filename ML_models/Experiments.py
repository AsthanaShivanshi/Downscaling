import sys
sys.path.append("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/")
from UNet import UNet
import torch
import torch.nn as nn
from Train import train_model
from Train import checkpoint_save
import os

def run_experiment(train_dataset, val_dataset, quick_test=False, num_epochs=30):
    model = UNet(in_channels=2, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if quick_test:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, range(100)),
            batch_size=64, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_dataset, range(30)), #Using a third of the samples for validation because I need to get a realistic picture according to the train test val split samples I actually have 
            batch_size=64, shuffle=False
        )
        num_epochs = 20
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        quick_test=quick_test
    )

    #### Saving the model is optional uncomment if requiredd
    final_val_loss= history['val_loss'][-1]
    checkpoint_save(model, optimizer, epoch=num_epochs, loss= final_val_loss, path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/checkpoints/All_Samples_64BS_30epochs_model_checkpoint_UNet_01.pth")
    return trained_model, history, final_val_loss
