import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Optional for progress bars

def train_one_epoch(model, dataloader, optimizer, criterion, device, quick_test=False):
    model.train()
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if quick_test and i == 2:
            break

    return running_loss / (i + 1)


def validate(model, dataloader, criterion, device, quick_test=False):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            if quick_test and j == 2:
                break

    return running_loss / (j + 1)


def train_model(
    model, train_loader, val_loader,
    optimizer, criterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs=50,
    quick_test=False
):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, quick_test)
        val_loss = validate(model, val_loader, criterion, device, quick_test)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, history

def checkpoint_save(model, optimizer, epoch, loss, path="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Trained_Models/UNet_model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

