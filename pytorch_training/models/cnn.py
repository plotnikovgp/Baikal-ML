from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import GradientReversal


class Conv1DBlock(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        return self.activation(out + residual)


class CNNModel(nn.Module):
    """CNN model for sequence processing."""

    def __init__(
        self,
        in_features: int = 5,
        hidden_size: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        out_size: int = 3,
        use_dilated_convolutions: bool = False,
        aggregate_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.aggregate_output = aggregate_output

        self.input_proj = nn.Linear(in_features, hidden_size)

        self.cnn_blocks = nn.ModuleList(
            [
                Conv1DBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**i if use_dilated_convolutions else 1,
                    dropout=dropout_p,
                )
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_size, out_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        for block in self.cnn_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        output = self.output_proj(x)

        if self.aggregate_output:
            output = output.mean(dim=1)

        return output


class CNNModelWithAttention(nn.Module):
    """CNN model with attention layers."""

    def __init__(
        self,
        in_features: int = 5,
        hidden_size: int = 128,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout_p: float = 0.1,
        out_size: int = 3,
        num_heads: int = 1,
        attention_layers: list[int] | None = None,
        aggregate_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if attention_layers is None:
            attention_layers = [1, 4]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.attention_layers = attention_layers
        self.aggregate_output = aggregate_output

        self.input_proj = nn.Linear(in_features, hidden_size)

        self.cnn_blocks = nn.ModuleList(
            [
                Conv1DBlock(hidden_size, hidden_size, kernel_size=kernel_size, dropout=dropout_p)
                for _ in range(num_layers)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_p, batch_first=True)
                for _ in range(len(attention_layers))
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(len(attention_layers))]
        )

        self.output_proj = nn.Linear(hidden_size, out_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        attn_idx = 0

        for i, block in enumerate(self.cnn_blocks):
            x = x.transpose(1, 2)
            x = block(x)
            x = x.transpose(1, 2)

            if i in self.attention_layers:
                x_norm = self.layer_norms[attn_idx](x)
                attn_output, _ = self.attention_blocks[attn_idx](x_norm, x_norm, x_norm)
                x = x + attn_output
                attn_idx += 1

        output = self.output_proj(x)

        if self.aggregate_output:
            output = output.mean(dim=1)

        return output


class CNNDomainAdaptation(nn.Module):
    """CNN with domain adaptation using gradient reversal."""

    def __init__(
        self,
        num_domains: int = 2,
        domain_classifier_hidden_size: int = 128,
        domain_classifier_layers: int = 2,
        gradient_reversal_alpha: float = 1.0,
        use_attention: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cnn = CNNModelWithAttention(**kwargs) if use_attention else CNNModel(**kwargs)

        if not self.cnn.aggregate_output:
            self.cnn.aggregate_output = True

        self.gradient_reversal = GradientReversal(alpha=gradient_reversal_alpha)
        self.domain_classifier = self._build_domain_classifier(
            self.cnn.hidden_size,
            domain_classifier_hidden_size,
            domain_classifier_layers,
            num_domains,
            self.cnn.dropout_p,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_domain_classifier(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_domains: int,
        dropout_p: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_size = input_size

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                ]
            )
            current_size = hidden_size

        layers.append(nn.Linear(current_size, num_domains))
        return nn.Sequential(*layers)

    def _get_features(self, x: Tensor) -> Tensor:
        features = self.cnn.input_proj(x)

        if isinstance(self.cnn, CNNModelWithAttention):
            attn_idx = 0
            for i, block in enumerate(self.cnn.cnn_blocks):
                features = features.transpose(1, 2)
                features = block(features)
                features = features.transpose(1, 2)

                if i in self.cnn.attention_layers:
                    x_norm = self.cnn.layer_norms[attn_idx](features)
                    attn_output, _ = self.cnn.attention_blocks[attn_idx](x_norm, x_norm, x_norm)
                    features = features + attn_output
                    attn_idx += 1
        else:
            features = features.transpose(1, 2)
            for block in self.cnn.cnn_blocks:
                features = block(features)
            features = features.transpose(1, 2)

        return features

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features = self._get_features(x)
        pooled_features = features.mean(dim=1)

        main_output = self.cnn.output_proj(pooled_features)

        reversed_features = self.gradient_reversal(pooled_features)
        domain_output = self.domain_classifier(reversed_features)

        return main_output, domain_output
