import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ik_model import IKNet


def load_csv_dataset(csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from CSV.

    Expected columns (order):
        Px, Py, Pz, Rx, Ry, Rz, theta1, theta2, theta3, theta4, theta5
    If headers exist with these names (case-insensitive), they will be used
    to select columns; otherwise the first 11 columns are assumed in order.
    """
    df = pd.read_csv(csv_path)
    lower_cols = [c.lower() for c in df.columns]

    def pick(names):
        idxs = []
        for n in names:
            if n.lower() in lower_cols:
                idxs.append(lower_cols.index(n.lower()))
        return idxs

    input_names = ["Px", "Py", "Pz", "Rx", "Ry", "Rz"]
    output_names = ["theta1", "theta2", "theta3", "theta4", "theta5"]
    x_idx = pick(input_names)
    y_idx = pick(output_names)

    if len(x_idx) == 6 and len(y_idx) == 5:
        X = df.iloc[:, x_idx].to_numpy(dtype=np.float32)
        Y = df.iloc[:, y_idx].to_numpy(dtype=np.float32)
    else:
        # Fallback: use first 6 for inputs and next 5 for outputs
        data = df.iloc[:, :11].to_numpy(dtype=np.float32)
        X, Y = data[:, :6], data[:, 6:11]

    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    return X_tensor, Y_tensor


def create_dummy_dataset(num_samples: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(42)
    X = rng.uniform(low=-1.0, high=1.0, size=(num_samples, 6)).astype(np.float32)
    # A toy mapping to generate continuous outputs similar to joint angles
    Y = np.stack([
        0.8 * X[:, 0] + 0.2 * X[:, 1] - 0.1 * X[:, 2],
        -0.5 * X[:, 3] + 0.3 * X[:, 4] + 0.2 * X[:, 5],
        0.7 * X[:, 0] - 0.4 * X[:, 4] + 0.1 * X[:, 2],
        0.6 * X[:, 1] + 0.4 * X[:, 3] - 0.2 * X[:, 5],
        -0.3 * X[:, 2] + 0.6 * X[:, 5],
    ], axis=1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def standardize(train_tensor: torch.Tensor, test_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_tensor - mean) / std, (test_tensor - mean) / std, mean, std


def main():
    parser = argparse.ArgumentParser(description="Train IK neural network")
    parser.add_argument("--csv", type=str, default="", help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--no_std", action="store_true", help="Disable input standardization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.csv:
        X, Y = load_csv_dataset(Path(args.csv))
    else:
        X, Y = create_dummy_dataset(4000)

    # Train/val split
    dataset = TensorDataset(X, Y)
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # Optional standardization on inputs only (fit on train split)
    if not args.no_std:
        X_train = train_ds.dataset.tensors[0][train_ds.indices]
        X_val = val_ds.dataset.tensors[0][val_ds.indices]
        X_train_std, X_val_std, mean, std = standardize(X_train, X_val)
        # Replace tensors in underlying dataset
        X_all = dataset.tensors[0]
        X_all[train_ds.indices] = X_train_std
        X_all[val_ds.indices] = X_val_std

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device(args.device)
    model = IKNet().to(device)
    criterion = nn.MSELoss()  # regression, Table 4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam, lr=0.001

    best_val = float("inf")
    save_path = Path("ik_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / train_len

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_running += loss.item() * xb.size(0)
        val_loss = val_running / max(1, val_len)

        tqdm.write(f"Epoch {epoch:4d}/{args.epochs} - train MSE: {train_loss:.6f} - val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "best_val_mse": best_val,
                "input_dim": 6,
                "output_dim": 5,
            }, save_path)

    print(f"Training complete. Best val MSE: {best_val:.6f}. Saved to {save_path}.")


if __name__ == "__main__":
    main()

