from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model import build_inverse_kinematics_model, compile_model
from .data import load_dataset, split_and_scale, INPUT_COLUMNS, OUTPUT_COLUMNS


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train inverse kinematics neural network")
	parser.add_argument("--data", required=True, help="Path to CSV dataset")
	parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (default: 2000)")
	parser.add_argument("--batch", type=int, default=64, help="Batch size (default: 64)")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
	parser.add_argument("--out", default="artifacts", help="Directory to save model and scalers")
	parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
	if args is None:
		args = parse_args()

	# Reproducibility
	seed = int(args.seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	# TensorFlow reproducibility settings
	keras.utils.set_random_seed(seed)

	X, y = load_dataset(args.data, INPUT_COLUMNS, OUTPUT_COLUMNS)
	(
		X_train, y_train,
		X_val, y_val,
		X_test, y_test,
		X_scaler, y_scaler,
	) = split_and_scale(X, y, scaler_dir=args.out)

	model = build_inverse_kinematics_model(input_dim=X.shape[1], output_dim=y.shape[1])
	model = compile_model(model, learning_rate=float(args.lr))

	callbacks = [
		keras.callbacks.EarlyStopping(
			monitor="val_loss", patience=int(args.patience), restore_best_weights=True
		),
		keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(args.out, "best_model.keras"),
			save_best_only=True,
			monitor="val_loss",
		),
	]

	history = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=int(args.epochs),
		batch_size=int(args.batch),
		callbacks=callbacks,
		verbose=2,
	)

	# Evaluate on test split
	test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
	print({k: float(v) for k, v in test_metrics.items()})

	os.makedirs(args.out, exist_ok=True)
	model.save(os.path.join(args.out, "final_model.keras"))

	# Save column metadata
	with open(os.path.join(args.out, "columns.txt"), "w", encoding="utf-8") as f:
		f.write("inputs," + ",".join(INPUT_COLUMNS) + "\n")
		f.write("outputs," + ",".join(OUTPUT_COLUMNS) + "\n")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

