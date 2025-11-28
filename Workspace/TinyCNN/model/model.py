import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """
    Input window: (B, 1, 6, 50), 6 channels (acc+gyro), 50 timesteps @100Hz 
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Temporal conv (1 x 16), 8 filters
        self.conv_t = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 16),
            stride=1,
            padding='same'  # preserved in the paper
        )
        # Spatial conv (6 x 1), 6 sensor channels, 8 filters
        self.conv_s = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=(6, 1),
            stride=1,
            padding='same'  # preserved in the paper
        )

        # GAP
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1:] == (6, 50):
            x = x.unsqueeze(1)  # (B, 1, 6, 50)
        elif x.dim() == 4 and x.shape[1:] == (1, 6, 50):
            pass
        else:
            raise ValueError(f"Expected input shape (B,6,50) or (B,1,6,50), got {tuple(x.shape)}")

        x = F.relu(self.conv_t(x))
        x = F.relu(self.conv_s(x))

        # Global Average Pooling -> (B, 8, 1, 1)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (B, 8)

        # Logits
        x = self.fc(x)
        return x
