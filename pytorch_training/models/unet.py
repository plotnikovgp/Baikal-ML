from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EncoderUNetBlock(nn.Module):
    """Encoder block with residual connection and downsampling."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()

        in_channels = in_channels if in_channels is not None else filters

        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv_downsample = nn.Conv1d(
            filters, filters, kernel_size, stride=2, padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(filters)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x1 = F.gelu(self.bn1(self.conv1(x))) * mask
        x2 = F.gelu(self.bn2(self.conv2(x1))) * mask
        x3 = F.gelu(self.bn3(self.conv_downsample(x2 + x1)))

        mask_downsampled = F.max_pool1d(mask, kernel_size=2, stride=2, padding=0)

        if mask_downsampled.size(2) != x3.size(2):
            diff = x3.size(2) - mask_downsampled.size(2)
            if diff > 0:
                mask_downsampled = F.pad(mask_downsampled, (0, diff))
            else:
                mask_downsampled = mask_downsampled[:, :, : x3.size(2)]

        return x3 * mask_downsampled, mask_downsampled


class DecoderUNetBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        in_filters: int | None = None,
        skip_filters: int | None = None,
    ) -> None:
        super().__init__()

        in_filters = in_filters if in_filters is not None else filters

        self.conv1 = nn.Conv1d(in_filters, filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv_upsample = nn.ConvTranspose1d(
            filters, filters, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1
        )
        self.bn3 = nn.BatchNorm1d(filters)

    def forward(
        self,
        x: Tensor,
        skip_x: Tensor,
        skip_mask: Tensor,
        next_mask: Tensor,
    ) -> Tensor:
        x1 = F.gelu(self.bn1(self.conv1(x))) * skip_mask
        x2 = F.gelu(self.bn2(self.conv2(x1))) * skip_mask
        x3 = F.gelu(self.bn3(self.conv_upsample(x2 + x1)))

        if x3.size(2) != skip_x.size(2):
            diff = skip_x.size(2) - x3.size(2)
            if diff > 0:
                x3 = F.pad(x3, (0, diff))
            else:
                x3 = x3[:, :, : skip_x.size(2)]

        if next_mask.size(2) != x3.size(2):
            diff = x3.size(2) - next_mask.size(2)
            if diff > 0:
                next_mask = F.pad(next_mask, (0, diff))
            else:
                next_mask = next_mask[:, :, : x3.size(2)]

        return torch.cat([x3 * next_mask, skip_x], dim=1)


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        filters: list[int],
        kernels: list[int],
        first_block_in_channels: int | None = None,
    ) -> None:
        super().__init__()

        assert len(filters) == len(kernels)

        blocks = [EncoderUNetBlock(filters[0], kernels[0], in_channels=first_block_in_channels)]
        for i in range(1, len(filters)):
            blocks.append(EncoderUNetBlock(filters[i], kernels[i], in_channels=filters[i - 1]))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        encodings = [x]
        masks = [mask]

        for block in self.blocks:
            x, mask = block(x, mask)
            encodings.append(x)
            masks.append(mask)

        return encodings, masks


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        filters: list[int],
        kernels: list[int],
        enc_filters: list[int] | None = None,
    ) -> None:
        super().__init__()

        assert len(filters) == len(kernels)

        self.blocks = nn.ModuleList(
            [
                DecoderUNetBlock(filters[0], kernels[0], in_filters=48),
                DecoderUNetBlock(filters[1], kernels[1], in_filters=96 + 96),
                DecoderUNetBlock(filters[2], kernels[2], in_filters=112 + 80),
            ]
        )

    def forward(self, encodings: list[Tensor], masks: list[Tensor]) -> Tensor:
        x = encodings[0]

        for i, block in enumerate(self.blocks):
            x = block(x, encodings[i + 1], masks[i], masks[i + 1])

        return x


class UNetModel(nn.Module):
    """U-Net model with bidirectional LSTM pre/post processing."""

    def __init__(
        self,
        pre_lstm_units: int,
        post_lstm_units: int,
        enc_filters: list[int],
        enc_kernels: list[int],
        dec_filters: list[int],
        dec_kernels: list[int],
        last_kernel: int,
        input_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pre_lstm_units = pre_lstm_units
        self.post_lstm_units = post_lstm_units

        self.pre_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=pre_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        self.encoder = UNetEncoder(enc_filters, enc_kernels, first_block_in_channels=pre_lstm_units)
        self.decoder = UNetDecoder(dec_filters, dec_kernels, enc_filters=enc_filters)

        self.post_lstm = nn.LSTM(
            input_size=96 + 64,
            hidden_size=post_lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        self.final_conv = nn.Conv1d(post_lstm_units, 2, last_kernel, padding="same")

    def _process_lstm(
        self,
        lstm: nn.LSTM,
        x: Tensor,
        seq_lengths: Tensor,
        units: int,
    ) -> Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        forward_out = out[:, :, :units]
        backward_out = out[:, :, units:]
        return forward_out * backward_out

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        mask = mask.unsqueeze(-1).float()
        seq_lengths = mask.sum(dim=1).squeeze(-1).cpu().int()
        conv_mask = mask.transpose(1, 2)

        lstm_out = self._process_lstm(self.pre_lstm, x, seq_lengths, self.pre_lstm_units)

        conv_input = lstm_out.transpose(1, 2) * conv_mask
        encodings, masks = self.encoder(conv_input, conv_mask)
        decoder_output = self.decoder(list(reversed(encodings)), list(reversed(masks)))

        decoder_output = decoder_output.transpose(1, 2)
        post_lstm = self._process_lstm(
            self.post_lstm, decoder_output, seq_lengths, self.post_lstm_units
        )

        conv_output = post_lstm.transpose(1, 2) * conv_mask
        logits = self.final_conv(conv_output)

        preds = torch.zeros_like(logits)
        preds[:, 1, :] = 1.0

        softmaxed = F.softmax(logits, dim=1)
        preds = torch.where(conv_mask.bool(), softmaxed, preds)

        return preds.transpose(1, 2)


# Backward compatibility alias
Encoder = UNetEncoder
Decoder = UNetDecoder
