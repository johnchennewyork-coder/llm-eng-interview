"""
=================================================================
  PYTORCH DEBUG CHALLENGE — SOLUTION
=================================================================
All 8 bugs fixed.  Each fix is tagged [FIX N] with an
explanation of the bug and its consequences.

Bug summary
-----------
  1. nn.Softmax + CrossEntropyLoss  (Convergence)
  2. Missing optimizer.zero_grad()  (Convergence)
  3. Appending loss tensor          (Memory leak)
  4. Missing model.eval()           (Correctness)
  5. Missing torch.no_grad()        (Performance / Memory)
  6. Hardcoded CPU device           (Performance)
  7. Training shuffle=False         (Convergence)
  8. Learning rate = 0.5            (Convergence)

=================================================================
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------
# Dataset — identical to the problem file (no bugs here)
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
            # ──────────────────────────────────────────────────────
            # [FIX 1] REMOVED nn.Softmax(dim=1)
            #
            # nn.CrossEntropyLoss internally computes
            #     log_softmax(input) then NLL.
            # Placing an explicit Softmax here means the pipeline
            # actually computes  log(softmax(softmax(logits))),
            # which compresses the output distribution and makes
            # gradients vanishingly small → the model cannot learn.
            # ──────────────────────────────────────────────────────
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

        # ──────────────────────────────────────────────────────
        # [FIX 2] Zero gradients BEFORE the forward pass.
        #
        # Without this call, gradients from every previous batch
        # are summed into the current gradient buffers.  The
        # effective gradient grows without bound, causing wild
        # parameter updates and divergence.
        # ──────────────────────────────────────────────────────
        optimizer.zero_grad()

        # Forward
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward
        loss.backward()
        optimizer.step()

        # ──────────────────────────────────────────────────────
        # [FIX 3] Extract a plain Python float with .item().
        #
        # Appending the raw loss *tensor* keeps every backward
        # graph alive in memory until the list is garbage-
        # collected at the end of the epoch.  For large models
        # this causes OOM; here it is simply wasteful.
        # ──────────────────────────────────────────────────────
        running_loss.append(loss.item())
        _, preds = outputs.max(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = sum(running_loss) / len(running_loss)
    return avg_loss, correct / total


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------
def validate(model, dataloader, criterion, device):
    """Evaluate on the validation set.  Returns (avg_loss, accuracy)."""

    # ──────────────────────────────────────────────────────
    # [FIX 4] Switch to evaluation mode.
    #
    # In training mode, Dropout randomly zeroes activations
    # and BatchNorm uses per-batch statistics.  This adds
    # noise to validation metrics and makes them unreliable.
    # model.eval() disables Dropout and tells BatchNorm to
    # use its accumulated running mean/variance.
    # ──────────────────────────────────────────────────────
    model.eval()

    running_loss = []
    correct = 0
    total = 0

    # ──────────────────────────────────────────────────────
    # [FIX 5] Disable gradient tracking during inference.
    #
    # Without torch.no_grad(), PyTorch builds a full backward
    # graph for every forward pass in the validation loop.
    # This wastes memory and compute — especially on GPU where
    # intermediate activations can be large.
    # ──────────────────────────────────────────────────────
    with torch.no_grad():
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
    print("  Spiral Classifier — Training Pipeline  (SOLUTION)")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # [FIX 6] Select the best available accelerator.
    #
    # The original code hardcoded torch.device("cpu"),
    # ignoring any CUDA or Apple-Silicon (MPS) GPU.
    # On a machine with a GPU this makes training orders
    # of magnitude slower than necessary.
    # ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    X_train, y_train, X_val, y_val = create_spiral_dataset()
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    # ──────────────────────────────────────────────────────
    # [FIX 7] Enable shuffle for the *training* DataLoader.
    #
    # Without shuffling, the model sees the same sample
    # ordering every epoch.  If the dataset has any
    # structure in its ordering (e.g. sorted by class)
    # this severely degrades convergence and generalisation.
    # ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0
    )

    # Model / Loss / Optimizer
    model = SpiralNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # ──────────────────────────────────────────────────────
    # [FIX 8] Use a sensible learning rate for Adam.
    #
    # Adam's recommended default is lr=1e-3.  A value of
    # 0.5 causes massive parameter updates that overshoot
    # every minimum, leading to divergence or oscillation.
    # ──────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
