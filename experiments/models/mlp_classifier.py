"""
MLP-based gesture classifier.
Flattens the sequence and uses fully connected layers.
Extremely fast inference but may lose some temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Simple MLP that flattens the sequence.

    Pros:
    - Extremely fast inference (~10x faster than TCN)
    - Very few operations
    - Good for real-time applications

    Cons:
    - Loses explicit temporal structure
    - May need more data to learn temporal patterns implicitly

    Parameters: ~100K (depends on sequence length)
    """

    def __init__(self, input_size: int, num_classes: int, seq_len: int = 15, dropout: float = 0.4):
        super().__init__()

        flat_size = input_size * seq_len
        hidden_size = 256

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        return self.net(x)


class TemporalMLP(nn.Module):
    """
    MLP with temporal awareness - processes each timestep then aggregates.
    Slightly slower than pure MLP but better at capturing patterns.
    """

    def __init__(self, input_size: int, num_classes: int, seq_len: int = 15, dropout: float = 0.3):
        super().__init__()

        # Per-timestep processing
        self.timestep_encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Temporal aggregation via learned weights
        self.temporal_weights = nn.Parameter(torch.ones(seq_len) / seq_len)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape

        # Encode each timestep
        encoded = self.timestep_encoder(x)  # (batch, seq_len, 64)

        # Weighted average over time (learned weights)
        weights = F.softmax(self.temporal_weights, dim=0)
        aggregated = (encoded * weights.view(1, -1, 1)).sum(dim=1)  # (batch, 64)

        return self.classifier(aggregated)


class StatisticalMLP(nn.Module):
    """
    MLP that uses statistical features (mean, std, min, max) over time.
    Very fast and robust to sequence length variations.
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # 4 statistics per feature: mean, std, min, max
        stat_size = input_size * 4

        self.net = nn.Sequential(
            nn.Linear(stat_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)

        # Compute statistics over time dimension
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        min_val = x.min(dim=1).values
        max_val = x.max(dim=1).values

        # Concatenate statistics
        stats = torch.cat([mean, std, min_val, max_val], dim=1)

        return self.net(stats)
