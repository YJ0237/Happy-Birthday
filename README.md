IK Neural Network (Inverse Kinematics)

This project provides a runnable PyTorch implementation of the inverse-kinematics neural network described in your figures:

- Inputs: 6 (Px, Py, Pz, Rx, Ry, Rz)
- Hidden layers: 48 → 120 → 60 neurons
- Dropout: 0.05 on the last two hidden layers
- Activations: sigmoid for hidden layers, linear for output
- Outputs: 5 (θ1..θ5)
- Optimizer: Adam, learning rate 0.001
- Loss: Mean Squared Error (MSE)
- Epochs: 2000

Files:
- `ik_model.py`: model definition
- `train_ik.py`: training script (loads CSV or generates dummy data)

Usage:
```bash
pip install -r requirements.txt
python train_ik.py               # trains on dummy data
python train_ik.py --csv your_dataset.csv  # train on your CSV
```

CSV format:
- Preferred header names: `Px,Py,Pz,Rx,Ry,Rz,theta1,theta2,theta3,theta4,theta5`
- If headers differ, the script falls back to using the first 11 columns in that order.

中文简要说明：
- 该工程实现了 6→48→120→60→5 的逆运动学网络；隐藏层使用 sigmoid，输出层线性；最后两层隐藏层使用 0.05 的 dropout；优化器 Adam(lr=0.001)，损失为 MSE，训练 2000 轮。
- `python train_ik.py --csv 数据集.csv` 可在自有数据上训练；若不提供 CSV，会自动生成可运行的虚拟数据。