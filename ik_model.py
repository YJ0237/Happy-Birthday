import torch
import torch.nn as nn


class IKNet(nn.Module):
    """Inverse-kinematics neural network.

    Architecture (from the paper description):
    - Input: 6 features (P_x, P_y, P_z, R_x, R_y, R_z)
    - Hidden1: 48 neurons, sigmoid
    - Hidden2: 120 neurons, sigmoid + Dropout(0.05)
    - Hidden3: 60 neurons, sigmoid + Dropout(0.05)
    - Output: 5 neurons (theta_1..theta_5), linear activation
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 5, dropout_p: float = 0.05):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.Sigmoid(),

            nn.Linear(48, 120),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_p),

            nn.Linear(120, 60),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_p),

            nn.Linear(60, output_dim),  # linear output for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(device: torch.device | None = None) -> IKNet:
    model = IKNet()
    if device is not None:
        model.to(device)
    return model


if __name__ == "__main__":
    # Quick shape sanity check
    net = IKNet()
    dummy = torch.zeros(2, 6)
    out = net(dummy)
    print("Output shape:", out.shape)

