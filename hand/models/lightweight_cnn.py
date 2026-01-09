"""
Lightweight 1D CNN for gesture classification.
Much simpler than TCN - uses standard convolutions without dilations.
Faster inference, fewer parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """
    Simple 1D CNN for gesture classification.

    Architecture:
    - 3 conv layers with increasing channels
    - Global average pooling (no attention overhead)
    - Single linear classifier

    Expected speedup: 2-3x faster than TCN
    Parameters: ~50K (vs ~200K for TCN)
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Convolutional layers - much smaller than TCN
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout)

        # Simple classifier
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling - much faster than attention
        x = x.mean(dim=2)  # (batch, 64)

        x = self.dropout(x)
        return self.classifier(x)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        b, c, _ = x.shape
        squeeze = x.mean(dim=2)  # (batch, channels)
        excite = self.fc(squeeze).view(b, c, 1)
        return x * excite


class LightweightCNNv3(nn.Module):
    """
    Conservative improvement over LightweightCNN:
    - Same structure but with combined avg+max pooling
    - Slightly wider channels (72, 144, 72 vs 64, 128, 64)

    Parameters: ~115K
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Slightly wider channels
        self.conv1 = nn.Conv1d(input_size, 72, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(72)

        self.conv2 = nn.Conv1d(72, 144, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(144)

        self.conv3 = nn.Conv1d(144, 72, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(72)

        self.dropout = nn.Dropout(dropout)

        # Combined avg+max pooling doubles features
        self.classifier = nn.Linear(72 * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Combined pooling
        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2).values
        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.dropout(x)
        return self.classifier(x)


class LightweightCNNv2(nn.Module):
    """
    Slightly larger variant with residual connection.
    Better accuracy while still being fast.
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        hidden_dim = 96

        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_dim, kernel_size=1)

        # Conv blocks with residual
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # Project to hidden dim
        x = self.input_proj(x)

        # Residual block 1
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)

        # Global average pooling
        x = x.mean(dim=2)

        x = self.dropout(x)
        return self.classifier(x)
