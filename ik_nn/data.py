from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


INPUT_COLUMNS = ["Px", "Py", "Pz", "Rx", "Ry", "Rz"]
OUTPUT_COLUMNS = ["theta1", "theta2", "theta3", "theta4", "theta5"]


def load_dataset(csv_path: str,
				 input_columns=INPUT_COLUMNS,
				 output_columns=OUTPUT_COLUMNS) -> Tuple[np.ndarray, np.ndarray]:
	"""Load dataset from CSV. The CSV must contain six input columns and five outputs.

	Args:
		csv_path: Path to CSV file.
		input_columns: Names of input columns.
		output_columns: Names of output columns.

	Returns:
		Tuple of (X, y) as numpy arrays.
	"""
	frame = pd.read_csv(csv_path)
	missing_inputs = [c for c in input_columns if c not in frame.columns]
	missing_outputs = [c for c in output_columns if c not in frame.columns]
	if missing_inputs or missing_outputs:
		raise ValueError(
			f"CSV missing columns. Missing inputs: {missing_inputs}; Missing outputs: {missing_outputs}"
		)
	X = frame[input_columns].to_numpy(dtype=np.float32)
	y = frame[output_columns].to_numpy(dtype=np.float32)
	return X, y


def split_and_scale(X: np.ndarray,
					y: np.ndarray,
					test_size: float = 0.1,
					val_size: float = 0.1,
					random_state: int = 42,
					scaler_dir: Optional[str] = None):
	"""Split into train/val/test and scale inputs and outputs.

	We standardize inputs and outputs with separate StandardScalers. Scalers can be saved
	to disk for later inference.
	"""
	X_train_full, X_test, y_train_full, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)
	# From the remaining, carve out validation
	val_ratio_of_full = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train_full, y_train_full, test_size=val_ratio_of_full, random_state=random_state
	)

	X_scaler = StandardScaler()
	y_scaler = StandardScaler()

	X_train_scaled = X_scaler.fit_transform(X_train)
	X_val_scaled = X_scaler.transform(X_val)
	X_test_scaled = X_scaler.transform(X_test)

	y_train_scaled = y_scaler.fit_transform(y_train)
	y_val_scaled = y_scaler.transform(y_val)
	y_test_scaled = y_scaler.transform(y_test)

	if scaler_dir is not None:
		os.makedirs(scaler_dir, exist_ok=True)
		dump(X_scaler, os.path.join(scaler_dir, "x_scaler.joblib"))
		dump(y_scaler, os.path.join(scaler_dir, "y_scaler.joblib"))

	return (
		X_train_scaled, y_train_scaled,
		X_val_scaled, y_val_scaled,
		X_test_scaled, y_test_scaled,
		X_scaler, y_scaler,
	)


def load_scalers(scaler_dir: str):
	X_scaler = load(os.path.join(scaler_dir, "x_scaler.joblib"))
	y_scaler = load(os.path.join(scaler_dir, "y_scaler.joblib"))
	return X_scaler, y_scaler


__all__ = [
	"INPUT_COLUMNS",
	"OUTPUT_COLUMNS",
	"load_dataset",
	"split_and_scale",
	"load_scalers",
]

