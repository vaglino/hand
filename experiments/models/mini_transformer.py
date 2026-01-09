"""
Mini Transformer for gesture classification.
Uses self-attention to capture temporal dependencies.
More expressive than CNN but heavier computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MiniTransformer(nn.Module):
    """
    Lightweight transformer for gesture sequences.

    Architecture:
    - Single transformer encoder layer
    - Learned positional encoding
    - CLS token for classification

    Parameters: ~80K
    Better than TCN for capturing long-range dependencies.
    """

    def __init__(self, input_size: int, num_classes: int, seq_len: int = 15,
                 d_model: int = 64, nhead: int = 4, dropout: float = 0.3):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Single transformer encoder layer (lightweight)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.shape[0]

        # Project to d_model dimensions
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Transformer encoding
        x = self.encoder(x)

        # Use CLS token for classification
        cls_output = x[:, 0]  # (batch, d_model)

        return self.classifier(cls_output)


class TinyTransformer(nn.Module):
    """
    Even smaller transformer variant.
    Uses mean pooling instead of CLS token for simplicity.

    Parameters: ~40K
    Fastest transformer variant.
    """

    def __init__(self, input_size: int, num_classes: int, seq_len: int = 15,
                 d_model: int = 48, nhead: int = 4, dropout: float = 0.3):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Simple learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Lightweight self-attention (manual implementation for speed)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        x = self.input_proj(x) + self.pos_encoding

        # Self-attention
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.d_model)
        q, k, v = qkv.unbind(dim=2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_model)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, v)
        x = self.proj(x) + residual

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual

        # Mean pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        return self.classifier(x)
