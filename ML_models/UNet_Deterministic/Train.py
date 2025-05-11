import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 
import wandb

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None, config=None):
    model.train()
    running_loss = 0.0
    quick_test = config["experiment"].get("quick_test", False)

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient norm (L2) per batch
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()

        # Log learning rate after every 20 batches as well in addition to the epoch loss
        if i % 20 == 0:
            log_dict = {
                "train_loss_batch": loss.item(),
                "grad_norm": grad_norm
            }
            if scheduler:
                log_dict["lr"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)

        if quick_test and i == 2:
            break

    return running_loss / (i + 1)


def validate(model, dataloader, criterion, config=None):
    model.eval()
    running_loss = 0.0
    quick_test = config["experiment"].get("quick_test", False)

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            if quick_test and j == 2:
                break

    return running_loss / (j + 1)

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, config=None):
    train_cfg = config["train"]
    num_epochs = train_cfg.get("num_epochs", 30)
    checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pth")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, config)
        val_loss = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss,
            "val_loss_epoch": val_loss,
            "lr_epoch": scheduler.get_last_lr()[0] if scheduler else None
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_save(model, optimizer, epoch+1, val_loss, checkpoint_path)

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
