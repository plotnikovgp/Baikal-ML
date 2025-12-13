from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTM(nn.Module):
    """Bidirectional LSTM with convolutional output layer."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_size: int,
        dropout_p: float = 0.0,
        num_layers: int = 2,
        kernel_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.conv = nn.Conv1d(hidden_size * 2, out_size, kernel_size, padding="same")

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        output = self.conv(lstm_out).permute(0, 2, 1)
        return output
