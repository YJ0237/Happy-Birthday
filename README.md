## Neural Network-Based Inverse Kinematics (Fig. 7)

This repo contains a minimal, runnable TensorFlow/Keras implementation of the inverse kinematics neural network described:

- Input layer: 6 neurons (Px, Py, Pz, Rx, Ry, Rz)
- Hidden layers: 48 (sigmoid), 120 (sigmoid + Dropout 0.05), 60 (sigmoid + Dropout 0.05)
- Output layer: 5 neurons (theta1..theta5), linear activation
- Optimizer: Adam
- Loss: MSE
- Default epochs: 2000

### Quickstart (demo data)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --demo --epochs 50 --output ./artifacts
```

Artifacts saved to `./artifacts`:
- `ik_model.keras`: best model (val_loss)
- `x_scaler.pkl`, `y_scaler.pkl`: `StandardScaler` objects
- `training_curves.png`: training/validation curves

### Training with your CSV

Expected CSV columns (order):
```
Px,Py,Pz,Rx,Ry,Rz,theta1,theta2,theta3,theta4,theta5
```
Headers are optional, extra columns are ignored.

```bash
python train.py --csv /path/to/dataset.csv --epochs 2000 --output ./artifacts
```

### Inference example

```python
import numpy as np, joblib, tensorflow as tf
from tensorflow.keras.models import load_model

x_scaler = joblib.load('artifacts/x_scaler.pkl')
y_scaler = joblib.load('artifacts/y_scaler.pkl')
model = tf.keras.models.load_model('artifacts/ik_model.keras')

x = np.array([[0.1, -0.2, 0.3, 0.0, 1.0, -0.5]], dtype=np.float32)
x_s = x_scaler.transform(x)
y_pred_s = model.predict(x_s)
y_pred = y_scaler.inverse_transform(y_pred_s)
print(y_pred)
```

### Code structure
- `ik_nn/model.py`: model builder
- `train.py`: data loading/generation, scaling, training, saving
