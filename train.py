import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from ik_nn.model import build_inverse_kinematics_model


def load_csv_dataset(csv_path: str):
    """Load dataset from CSV.

    Expected columns (order): Px,Py,Pz,Rx,Ry,Rz,theta1,theta2,theta3,theta4,theta5
    Extra columns are ignored; named/unnamed headers are supported.
    """
    df = pd.read_csv(csv_path)
    values = df.values.astype(np.float32)
    if values.shape[1] < 11:
        raise ValueError(
            "CSV must have at least 11 columns: 6 inputs + 5 outputs."
        )
    X = values[:, :6]
    y = values[:, 6:11]
    return X, y


def generate_demo_data(num_samples: int = 5000, noise_std: float = 0.01, seed: int = 13):
    """Generate synthetic IK-like data for quick sanity training.

    This does NOT represent a real robot; it is for demonstration only.
    """
    rng = np.random.default_rng(seed)
    Px, Py, Pz = rng.uniform(-1.0, 1.0, (3, num_samples))
    Rx, Ry, Rz = rng.uniform(-np.pi, np.pi, (3, num_samples))
    X = np.stack([Px, Py, Pz, Rx, Ry, Rz], axis=1).astype(np.float32)

    # A simple smooth mapping to 5 joint-like outputs
    t1 = np.arctan2(Py, Px)
    t2 = np.sin(Rx) + 0.5 * Px
    t3 = np.cos(Ry) + 0.3 * Py
    t4 = np.sin(Rz) + 0.2 * Pz
    t5 = 0.3 * Px - 0.2 * Py + 0.1 * Pz + 0.5 * np.sin(Rx + Ry + Rz)
    Y = np.stack([t1, t2, t3, t4, t5], axis=1).astype(np.float32)

    Y += rng.normal(0.0, noise_std, Y.shape).astype(np.float32)
    return X, Y


def main():
    parser = argparse.ArgumentParser(description="Train IK NN model (Fig.7)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (Table 4)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (Table 4)")
    parser.add_argument("--output", type=str, default="./artifacts", help="Directory to save model and scalers")
    parser.add_argument("--demo", action="store_true", help="Use generated demo data instead of CSV")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.csv and not args.demo:
        X, Y = load_csv_dataset(args.csv)
    else:
        X, Y = generate_demo_data()

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    model = build_inverse_kinematics_model(learning_rate=args.learning_rate)
    model.summary(print_fn=lambda s: print(s))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, "ik_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6),
    ]

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    # Save scalers for inference
    import joblib

    joblib.dump(x_scaler, os.path.join(args.output, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(args.output, "y_scaler.pkl"))

    # Plot history
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.history["loss"], label="train")
        ax[0].plot(history.history["val_loss"], label="val")
        ax[0].set_title("MSE Loss")
        ax[0].legend()
        ax[1].plot(history.history["mae"], label="train")
        ax[1].plot(history.history["val_mae"], label="val")
        ax[1].set_title("MAE")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "training_curves.png"))
    except Exception as e:
        print(f"Skipping plots: {e}")


if __name__ == "__main__":
    main()