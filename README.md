## Neural Network-Based Inverse Kinematics (Fig. 7)

This project reproduces the neural network architecture and training setup described in the prompt:

- **Inputs (6):** `Px, Py, Pz, Rx, Ry, Rz`
- **Outputs (5):** `theta1 .. theta5`
- **Hidden layers:** 48 → 120 → 60 neurons
- **Dropout:** 0.05 after the last two hidden layers
- **Activations:** sigmoid in hidden layers, linear in output layer
- **Optimizer:** Adam, **learning rate:** 0.001
- **Loss:** Mean Squared Error (MSE)
- **Epochs:** default 2000

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data format

Provide a CSV with columns:

```
Px,Py,Pz,Rx,Ry,Rz,theta1,theta2,theta3,theta4,theta5
```

Example row:

```
0.12,0.05,0.30,0.00,1.57,0.00,-0.21,1.10,0.58,-0.33,0.75
```

### Train

```bash
python -m ik_nn.train --data /path/to/dataset.csv --out artifacts --epochs 2000 --lr 0.001
```

Artifacts saved to `artifacts/`:

- `best_model.keras`, `final_model.keras`
- `x_scaler.joblib`, `y_scaler.joblib`
- `columns.txt`

### Inference

```bash
python -m ik_nn.predict --model_dir artifacts --x 0.12 0.05 0.30 0.00 1.57 0.00
```

Output is a JSON-like dict with predicted joint angles.

### Notes

- Inputs and outputs are standardized during training and the scalers are saved for inference.
- The network strictly uses sigmoid activations for hidden layers and linear activation for outputs to match a regression setup.

# Happy-Birthday