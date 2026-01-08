import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Single temporal block with dilated convolution and residual connection."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # Calculate causal padding (only pad left side)
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation
        )  # No padding, we'll pad manually
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation
        )  # No padding, we'll pad manually
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        # Apply causal padding (pad only the left side)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # Apply causal padding again for second conv
        if self.padding > 0:
            out = F.pad(out, (self.padding, 0))

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Prepare residual - ensure same size as output
        if self.downsample is not None:
            # Apply same causal padding to input for residual
            if self.padding > 0:
                x_padded = F.pad(x, (self.padding, 0))
            else:
                x_padded = x
            res = self.downsample(x_padded)
            # Crop to match output size
            if res.size(2) > out.size(2):
                res = res[:, :, : out.size(2)]
        else:
            res = x
            # Crop residual to match output size if needed
            if res.size(2) > out.size(2):
                res = res[:, :, : out.size(2)]

        return F.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for gesture classification."""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EnhancedGestureClassifier(nn.Module):
    """Enhanced gesture classifier using TCN architecture."""

    def __init__(self, input_size, num_classes, dropout=0.25):
        super().__init__()

        # TCN layers with increasing receptive field
        tcn_channels = [64, 96, 128, 160]
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size=3, dropout=dropout)

        # Global attention pooling
        self.attention = nn.Sequential(nn.Linear(tcn_channels[-1], 64), nn.Tanh(), nn.Linear(64, 1))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)

        # TCN processing
        tcn_out = self.tcn(x)  # (batch_size, channels, seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch_size, seq_len, channels)

        # Attention pooling
        attention_weights = F.softmax(self.attention(tcn_out), dim=1)
        context = torch.sum(attention_weights * tcn_out, dim=1)

        # Classification
        return self.classifier(context)

