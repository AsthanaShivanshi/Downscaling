import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os
import wandb

def train_one_epoch(model, dataloader, optimizer, criterion, quick_test=False):
    model.train()
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if quick_test and i == 2:
            break

    return running_loss / (i + 1)


def validate(model, dataloader, criterion, quick_test=False):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            if quick_test and j == 2:
                break

    return running_loss / (j + 1)


def train_model(
    model, train_loader, val_loader,
    optimizer, criterion,
    num_epochs=30,
    quick_test=False
):
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, quick_test)
        val_loss = validate(model, val_loader, criterion, quick_test)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_save(
                model, optimizer, epoch+1, val_loss,
                path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/checkpoints/fullmodel_best_model_checkpoint.pth"
            )

    return model, history

def checkpoint_save(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Best model checkpoint saved at: {path}")
