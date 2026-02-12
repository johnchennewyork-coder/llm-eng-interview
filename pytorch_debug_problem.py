"""
=================================================================
  PYTORCH DEBUG CHALLENGE — HARD LEVEL
=================================================================

Scenario
--------
You've inherited a training pipeline from a colleague who left
the company. The model is a multi-class classifier trained on a
synthetic 4-class spiral dataset.

Expected behaviour:  >85 % validation accuracy within 20 epochs.
Actual behaviour:    The model barely learns, trains slowly, and
                     leaks memory.

Your task
---------
There are **8 bugs** hidden in this file.  They span four
categories:

  • Convergence — the model fails to learn properly
  • Performance — training is slower than it should be
  • Memory      — unnecessary GPU/CPU memory consumption
  • Correctness — evaluation produces wrong metrics

Find and fix every bug, then verify your work:

    python pytorch_debug_ui.py

=================================================================
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------
# Dataset — 4-class spiral (synthetic, no download needed)
# ----------------------------------------------------------------
def create_spiral_dataset(n_points=1000, n_classes=4, noise=0.2):
    """Return (X_train, y_train, X_val, y_val) tensors."""
    np.random.seed(42)
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)

    for cls in range(n_classes):
        idx = range(n_points * cls, n_points * (cls + 1))
        r = np.linspace(0.0, 1, n_points)
        t = (
            np.linspace(cls * 4, (cls + 1) * 4, n_points)
            + np.random.randn(n_points) * noise
        )
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = cls

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    return (
        torch.FloatTensor(X[:split]),
        torch.LongTensor(y[:split]),
        torch.FloatTensor(X[split:]),
        torch.LongTensor(y[split:]),
    )


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------
class SpiralNet(nn.Module):
    """3-layer MLP for spiral classification."""

    def __init__(self, input_dim=2, hidden_dim=128, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = []
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss.append(loss)
        _, preds = outputs.max(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = sum(running_loss) / len(running_loss)
    return avg_loss.item(), correct / total


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------
def validate(model, dataloader, criterion, device):
    """Evaluate on the validation set.  Returns (avg_loss, accuracy)."""
    running_loss = []
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        running_loss.append(loss.item())
        _, preds = outputs.max(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return sum(running_loss) / len(running_loss), correct / total


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Spiral Classifier — Training Pipeline")
    print("=" * 60)

    # Device
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    X_train, y_train, X_val, y_val = create_spiral_dataset()
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0
    )

    # Model / Loss / Optimizer
    model = SpiralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.5)

    # Training loop
    n_epochs = 20
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(n_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time"].append(dt)

        print(
            f"Epoch {epoch+1:2d}/{n_epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
            f"time={dt:.3f}s"
        )

    print("=" * 60)
    print(
        f"  Final  train_acc={history['train_acc'][-1]:.4f}  "
        f"val_acc={history['val_acc'][-1]:.4f}"
    )
    print(f"  Avg epoch time: {np.mean(history['epoch_time']):.3f}s")
    print("=" * 60)

    return history


if __name__ == "__main__":
    main()
