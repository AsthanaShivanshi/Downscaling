import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os

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

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,quick_test)
        val_loss = validate(model, val_loader, criterion,quick_test)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

    return model, history

def checkpoint_save(model, optimizer, epoch, loss, path):

    if epoch%20==0 and epoch!=0:

        # Save the model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, path)

        print(f"Model checkpoint saved at: {path}")

