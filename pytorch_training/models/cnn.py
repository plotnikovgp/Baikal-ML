import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import GradientReversal


class Conv1DBlock(nn.Module):
    """
    Basic convolutional block with residual connection
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2 * dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2 * dilation,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: [B, C, N]
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.activation(out)

        return out


class CNNModel(nn.Module):
    def __init__(
        self,
        in_features=5,
        hidden_size=128,
        num_layers=4,
        kernel_size=3,
        dropout_p=0.1,
        out_size=3,
        use_dilated_convolutions=False,
        aggregate_output=False,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.aggregate_output = aggregate_output
        self.dropout_p = dropout_p

        self.input_proj = nn.Linear(in_features, hidden_size)

        self.cnn_blocks = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2**i if use_dilated_convolutions else 1

            self.cnn_blocks.append(
                Conv1DBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout_p,
                )
            )

        self.output_proj = nn.Linear(hidden_size, out_size)
        self._init_weights()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CNN Model initialized with {num_params/1e6:.3f}M parameters")

    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
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

    def forward(self, x, mask=None):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape [B, N, F]

        Returns:
            Output tensor of shape:
            - [B, N, out_size] if aggregate_output=False
            - [B, out_size] if aggregate_output=True
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)  # [B, N, hidden_size]

        # Transpose for Conv1D (expects [B, C, N])
        x = x.transpose(1, 2)  # [B, hidden_size, N]

        # Apply CNN blocks
        for block in self.cnn_blocks:
            x = block(x)

        # Transpose back to [B, N, hidden_size]
        x = x.transpose(1, 2)

        output = self.output_proj(x)  # [B, N, out_size]

        if self.aggregate_output:
            output = output.mean(dim=1)  # [B, out_size]

        return output


class CNNModelWithAttention(nn.Module):
    def __init__(
        self,
        in_features=5,
        hidden_size=128,
        num_layers=4,
        kernel_size=3,
        dropout_p=0.1,
        out_size=3,
        num_heads=1,
        attention_layers=[1, 3],  # Indices of layers where to add attention
        aggregate_output=False,
    ):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.attention_layers = attention_layers
        self.aggregate_output = aggregate_output

        self.input_proj = nn.Linear(in_features, hidden_size)

        self.cnn_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.cnn_blocks.append(
                Conv1DBlock(
                    hidden_size, hidden_size, kernel_size=kernel_size, dropout=dropout_p
                )
            )

        self.attention_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout_p, batch_first=True
                )
                for _ in range(len(attention_layers))
            ]
        )

        # Layer norms for attention
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(len(attention_layers))]
        )

        self.output_proj = nn.Linear(hidden_size, out_size)

        self._init_weights()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"CNN Model with Attention initialized with {num_params/1e6:.3f}M parameters"
        )

    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
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

    def forward(self, x, mask=None):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape [B, N, F]

        Returns:
            Output tensor of shape:
            - [B, N, out_size] if aggregate_output=False
            - [B, out_size] if aggregate_output=True
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)  # [B, N, hidden_size]

        attn_idx = 0

        for i, block in enumerate(self.cnn_blocks):
            x = x.transpose(1, 2)  # [B, hidden_size, N]
            x = block(x)
            x = x.transpose(1, 2)  # [B, N, hidden_size]

            if i in self.attention_layers:
                x_norm = self.layer_norms[attn_idx](x)

                # Apply self-attention
                attn_output, _ = self.attention_blocks[attn_idx](
                    query=x_norm, key=x_norm, value=x_norm
                )

                x = x + attn_output
                attn_idx += 1

        output = self.output_proj(x)  # [B, N, out_size]

        if self.aggregate_output:
            output = output.mean(dim=1)  # [B, out_size]

        return output


class CNNDomainAdaptation(nn.Module):
    def __init__(
        self,
        num_domains=2,
        domain_classifier_hidden_size=128,
        domain_classifier_layers=2,
        gradient_reversal_alpha=1.0,
        use_attention=False,
        **kwargs,
    ):
        super().__init__()

        # Create the appropriate CNN model
        if use_attention:
            self.cnn = CNNModelWithAttention(**kwargs)
        else:
            self.cnn = CNNModel(**kwargs)

        # Make sure the model aggregates outputs for domain adaptation
        if not self.cnn.aggregate_output:
            print("Warning: Setting aggregate_output=True for domain adaptation")
            self.cnn.aggregate_output = True

        # We don't need a separate main_head - we'll use the CNN's output_proj
        self.gradient_reversal = GradientReversal(alpha=gradient_reversal_alpha)

        # Build the domain classifier
        domain_classifier_layers_list = []
        input_size = self.cnn.hidden_size

        for _ in range(domain_classifier_layers - 1):
            domain_classifier_layers_list.extend(
                [
                    nn.Linear(input_size, domain_classifier_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.cnn.dropout_p),  # Use the CNN's dropout parameter
                ]
            )
            input_size = domain_classifier_hidden_size

        domain_classifier_layers_list.append(nn.Linear(input_size, num_domains))

        self.domain_classifier = nn.Sequential(*domain_classifier_layers_list)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"CNN Domain Adaptation Model initialized with {num_params/1e6:.3f}M parameters"
        )

    def forward(self, x, mask=None):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape [B, N, F]
            mask: Optional mask

        Returns:
            Tuple of (main_output, domain_output)
        """
        batch_size, seq_len, _ = x.shape

        if isinstance(self.cnn, CNNModelWithAttention):
            # Process with attention CNN
            features = self.cnn.input_proj(x)  # [B, N, hidden_size]
            attn_idx = 0

            for i, block in enumerate(self.cnn.cnn_blocks):
                features = features.transpose(1, 2)  # [B, hidden_size, N]
                features = block(features)
                features = features.transpose(1, 2)  # [B, N, hidden_size]

                if i in self.cnn.attention_layers:
                    x_norm = self.cnn.layer_norms[attn_idx](features)

                    # Apply self-attention
                    attn_output, _ = self.cnn.attention_blocks[attn_idx](
                        query=x_norm, key=x_norm, value=x_norm
                    )

                    features = features + attn_output
                    attn_idx += 1
        else:
            # Process with regular CNN
            features = self.cnn.input_proj(x)  # [B, N, hidden_size]
            features = features.transpose(1, 2)  # [B, hidden_size, N]

            # Apply CNN blocks
            for block in self.cnn.cnn_blocks:
                features = block(features)

            features = features.transpose(1, 2)  # [B, N, hidden_size]

        # At this point, features is [B, N, hidden_size]
        # Aggregate features
        pooled_features = features.mean(dim=1)  # [B, hidden_size]

        # Use the CNN's output projection for the main task
        main_output = self.cnn.output_proj(pooled_features)  # [B, out_size]

        # Apply domain adaptation
        reversed_features = self.gradient_reversal(pooled_features)
        domain_output = self.domain_classifier(reversed_features)  # [B, num_domains]

        return main_output, domain_output


def sample_run():
    """
    Create and test various CNN model configurations to show parameter counts
    """
    print("\n" + "=" * 80)
    print("CNN MODEL PARAMETER COUNTS DEMONSTRATION")
    print("=" * 80)

    batch_size = 16
    seq_length = 32
    in_features = 5
    out_size = 3

    x = torch.randn(batch_size, seq_length, in_features)

    print("\nCNNModel configurations:")
    print("-" * 60)
    print(
        f"{'Hidden Size':<12} {'Num Layers':<12} {'Parameters':<12} {'Parameter Count':<18}"
    )
    print("-" * 60)

    for hidden_size in [64, 128, 256, 512]:
        for num_layers in [2, 3, 5, 8]:
            # Create model
            model = CNNModel(
                in_features=in_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                out_size=out_size,
            )

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Forward pass to test
            with torch.no_grad():
                output = model(x)
                assert output.shape == (batch_size, seq_length, out_size)

            # Print parameter count
            print(
                f"{hidden_size:<12} {num_layers:<12} {num_params:<12,d} {num_params/1e6:.3f}M"
            )

    # Test CNNModelWithAttention with different configurations
    print("\nCNNModelWithAttention configurations:")
    print("-" * 80)
    print(
        f"{'Hidden Size':<12} {'Num Layers':<12} {'Num Heads':<12} {'Parameters':<12} {'Parameter Count':<18}"
    )
    print("-" * 80)

    for hidden_size in [64, 128, 256, 512]:
        for num_layers in [3, 5, 8]:
            for num_heads in [1]:
                # Create model
                model = CNNModelWithAttention(
                    in_features=in_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    out_size=out_size,
                    num_heads=num_heads,
                    attention_layers=[
                        1,
                        num_layers - 2,
                    ],  # Add attention after 2nd layer and 2nd-to-last layer
                )

                # Count parameters
                num_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                # Forward pass to test
                with torch.no_grad():
                    output = model(x)
                    assert output.shape == (batch_size, seq_length, out_size)

                # Print parameter count
                print(
                    f"{hidden_size:<12} {num_layers:<12} {num_heads:<12} {num_params:<12,d} {num_params/1e6:.3f}M"
                )

    # Demonstrate aggregated output
    print("\n" + "=" * 80)
    print("OUTPUT AGGREGATION DEMONSTRATION")
    print("=" * 80)

    # Create a standard model without aggregation
    standard_model = CNNModel(
        in_features=in_features,
        hidden_size=128,
        num_layers=4,
        out_size=out_size,
        aggregate_output=False,
    )

    # Create the same model with aggregation
    aggregated_model = CNNModel(
        in_features=in_features,
        hidden_size=128,
        num_layers=4,
        out_size=out_size,
        aggregate_output=True,
    )

    # Forward pass through both models
    with torch.no_grad():
        standard_output = standard_model(x)
        aggregated_output = aggregated_model(x)

    # Print output shapes
    print(f"\nStandard model output shape: {standard_output.shape}")
    print(f"Aggregated model output shape: {aggregated_output.shape}")


if __name__ == "__main__":
    # Run the sample function when the script is executed directly
    sample_run()
