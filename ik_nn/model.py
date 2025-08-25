import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_inverse_kinematics_model(input_dim: int = 6, output_dim: int = 5) -> keras.Model:

	inputs = keras.Input(shape=(input_dim,), name="ik_inputs")

	# Hidden Layer 1: 48 neurons, sigmoid
	x = layers.Dense(48, activation="sigmoid", name="hidden_1_dense48_sigmoid")(inputs)

	# Hidden Layer 2: 120 neurons, sigmoid + Dropout 0.05
	x = layers.Dense(120, activation="sigmoid", name="hidden_2_dense120_sigmoid")(x)
	x = layers.Dropout(0.05, name="hidden_2_dropout005")(x)

	# Hidden Layer 3: 60 neurons, sigmoid + Dropout 0.05
	x = layers.Dense(60, activation="sigmoid", name="hidden_3_dense60_sigmoid")(x)
	x = layers.Dropout(0.05, name="hidden_3_dropout005")(x)

	# Output: 5 neurons, linear activation
	outputs = layers.Dense(output_dim, activation="linear", name="theta_outputs")(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name="inverse_kinematics_nn")
	return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
	optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss="mse", metrics=[keras.metrics.MeanSquaredError(name="mse")])
	return model


__all__ = ["build_inverse_kinematics_model", "compile_model"]

