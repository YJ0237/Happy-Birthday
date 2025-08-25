import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_inverse_kinematics_model(
    learning_rate: float = 0.001,
    input_dim: int = 6,
    output_dim: int = 5,
    hidden_units: tuple[int, int, int] = (48, 120, 60),
    dropout_rate: float = 0.05,
) -> tf.keras.Model:
    """Create the inverse kinematics neural network as specified.

    Architecture (Fig.7):
      - Input: 6 features (Px, Py, Pz, Rx, Ry, Rz)
      - Hidden1: 48 units, sigmoid
      - Hidden2: 120 units, sigmoid + Dropout(0.05)
      - Hidden3: 60 units, sigmoid + Dropout(0.05)
      - Output: 5 units (theta1..theta5), linear activation

    Optimizer/Loss per Table 4: Adam + MSE.
    """

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(hidden_units[0], activation="sigmoid", name="hidden_1"),
            Dense(hidden_units[1], activation="sigmoid", name="hidden_2"),
            Dropout(dropout_rate, name="dropout_2"),
            Dense(hidden_units[2], activation="sigmoid", name="hidden_3"),
            Dropout(dropout_rate, name="dropout_3"),
            Dense(output_dim, activation="linear", name="output"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model