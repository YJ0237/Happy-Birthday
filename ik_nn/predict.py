from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
from tensorflow import keras

from .data import load_scalers, INPUT_COLUMNS, OUTPUT_COLUMNS


def parse_args():
	parser = argparse.ArgumentParser(description="Run inference for inverse kinematics model")
	parser.add_argument("--model_dir", required=True, help="Directory containing saved model and scalers")
	parser.add_argument(
		"--x",
		nargs=6,
		type=float,
		required=True,
		help="Six input values in order: Px Py Pz Rx Ry Rz",
	)
	return parser.parse_args()


def main(args=None):
	if args is None:
		args = parse_args()

	model_path = os.path.join(args.model_dir, "best_model.keras")
	if not os.path.exists(model_path):
		model_path = os.path.join(args.model_dir, "final_model.keras")

	model = keras.models.load_model(model_path, compile=False)
	X_scaler, y_scaler = load_scalers(args.model_dir)

	x_input = np.asarray(args.x, dtype=np.float32).reshape(1, -1)
	x_scaled = X_scaler.transform(x_input)
	y_scaled_pred = model.predict(x_scaled, verbose=0)
	y_pred = y_scaler.inverse_transform(y_scaled_pred)

	result = {name: float(val) for name, val in zip(OUTPUT_COLUMNS, y_pred.flatten())}
	print(result)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

