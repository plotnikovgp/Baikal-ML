import torch
import torch.nn as nn
import torch.nn.functional as F

# WARNING: dirty implementation, not tested, to be removed or rewritten

class EncoderUNetBlock(nn.Module):
    def __init__(self, filters, kernel_size, in_channels=None):
        super().__init__()
        in_channels = in_channels if in_channels is not None else filters
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv_downsample = nn.Conv1d(filters, filters, kernel_size, stride=2, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(filters)

    def forward(self, x, mask):
        x1 = self.conv1(x)
        x1 = F.gelu(self.bn1(x1))
        x1 = x1 * mask
        
        x2 = self.conv2(x1)
        x2 = F.gelu(self.bn2(x2))
        x2 = x2 * mask
        
        x3 = x2 + x1
        x3 = self.conv_downsample(x3)
        x3 = F.gelu(self.bn3(x3))
        
        mask_downsampled = F.max_pool1d(mask, kernel_size=2, stride=2, padding=0)
        
        if mask_downsampled.size(2) != x3.size(2):
            if mask_downsampled.size(2) < x3.size(2):
                mask_downsampled = F.pad(mask_downsampled, (0, x3.size(2) - mask_downsampled.size(2)))
            else:
                mask_downsampled = mask_downsampled[:, :, :x3.size(2)]
        
        encoding = x3 * mask_downsampled
        
        return encoding, mask_downsampled


class Encoder(nn.Module):
    def __init__(self, filters, kernels, first_block_in_channels=None):
        super().__init__()
        assert len(filters) == len(kernels)
        
        blocks = []
        blocks.append(EncoderUNetBlock(filters[0], kernels[0], in_channels=first_block_in_channels))
        
        for i in range(1, len(filters)):
            blocks.append(EncoderUNetBlock(filters[i], kernels[i], in_channels=filters[i-1]))
        
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask):
        encodings = [x]
        masks = [mask]
        
        for block in self.blocks:
            x, mask = block(x, mask)
            encodings.append(x)
            masks.append(mask)
            
        return encodings, masks


class DecoderUNetBlock(nn.Module):
    def __init__(self, filters, kernel_size, in_filters=None, skip_filters=None):
        super().__init__()
        self.in_filters = in_filters if in_filters is not None else filters
        self.skip_filters = skip_filters if skip_filters is not None else filters
        
        self.conv1 = nn.Conv1d(self.in_filters, filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv_upsample = nn.ConvTranspose1d(
            filters, filters, kernel_size, 
            stride=2, 
            padding=kernel_size//2,
            output_padding=1  # This ensures output size is exactly 2x the input size
        )
        self.bn3 = nn.BatchNorm1d(filters)

    def forward(self, x, skip_x, skip_mask, next_mask):
        x1 = self.conv1(x)
        x1 = F.gelu(self.bn1(x1))
        x1 = x1 * skip_mask
        
        x2 = self.conv2(x1)
        x2 = F.gelu(self.bn2(x2))
        x2 = x2 * skip_mask
        
        x3 = x2 + x1
        x3 = self.conv_upsample(x3)
        x3 = F.gelu(self.bn3(x3))
        
        # Ensure x3 has the same size as skip_x
        if x3.size(2) != skip_x.size(2):
            if x3.size(2) > skip_x.size(2):
                x3 = x3[:, :, :skip_x.size(2)]
            else:
                # Pad if necessary (shouldn't normally happen with proper transposed conv params)
                x3 = F.pad(x3, (0, skip_x.size(2) - x3.size(2)))
        
        if next_mask.size(2) != x3.size(2):
            if next_mask.size(2) < x3.size(2):
                next_mask = F.pad(next_mask, (0, x3.size(2) - next_mask.size(2)))
            else:
                next_mask = next_mask[:, :, :x3.size(2)]
                
        x3 = x3 * next_mask
        output = torch.cat([x3, skip_x], dim=1)
        
        return output


class Decoder(nn.Module):
    def __init__(self, filters, kernels, enc_filters=None):
        super().__init__()
        assert len(filters) == len(kernels)
        
        self.enc_filters = enc_filters if enc_filters is not None else filters
        
        self.blocks = nn.ModuleList([
            # channels configuration for some reason is differenct from tf implementation
            DecoderUNetBlock(filters[0], kernels[0], in_filters=48),
            DecoderUNetBlock(filters[1], kernels[1], in_filters=96 + 96),
            DecoderUNetBlock(filters[2], kernels[2], in_filters=112 + 80)
        ])

    def forward(self, encodings, masks):
        x = encodings[0]
        
        for i, block in enumerate(self.blocks):
            skip_x = encodings[i+1]
            skip_mask = masks[i]
            next_mask = masks[i+1]
            
            x = block(x, skip_x, skip_mask, next_mask)
            
        return x


class UNetModel(nn.Module):
    def __init__(self, pre_lstm_units, post_lstm_units, enc_filters, enc_kernels, 
                 dec_filters, dec_kernels, last_kernel, input_dim=None):
        super().__init__()
        self.pre_lstm_units = pre_lstm_units
        self.post_lstm_units = post_lstm_units
        self.enc_filters = enc_filters
        self.enc_kernels = enc_kernels
        self.dec_filters = dec_filters
        self.dec_kernels = dec_kernels
        self.last_kernel = last_kernel
        self.input_dim = input_dim
        
        self.pre_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=pre_lstm_units,
            batch_first=True,
            bidirectional=True
        )

        self.encoder = Encoder(enc_filters, enc_kernels, first_block_in_channels=pre_lstm_units)
        self.decoder = Decoder(dec_filters, dec_kernels, enc_filters=enc_filters)
        
        self.post_lstm = nn.LSTM(
            input_size=96 + 64,  # Output from decoder after last concatenation
            hidden_size=post_lstm_units,
            batch_first=True,
            bidirectional=True
        )
        
        self.final_conv = nn.Conv1d(post_lstm_units, 2, last_kernel, padding='same')

    def forward(self, x, mask):
        """
        Forward pass of the U-Net model
        
        Args:
            x: Input tensor [B, T, F]
            mask: Mask tensor [B, T, 1]
        
        Returns:
            preds: Prediction tensor [B, 2, T]
        """
        batch_size, seq_len = x.shape[:2]
                seq_lengths = mask.sum(dim=1).squeeze(-1).cpu().int()
        
        mask_bool = mask.squeeze(-1).bool()
        
        packed_features = nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths=seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_lstm_out, _ = self.pre_lstm(packed_features)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        forward_out = lstm_out[:, :, :self.pre_lstm_units]
        backward_out = lstm_out[:, :, self.pre_lstm_units:]
        lstm_out = forward_out * backward_out  # Element-wise multiplication to mimic TF's 'mul' merge mode
        
        # Transpose to channel-first for convolutions and apply mask
        conv_input = lstm_out.transpose(1, 2)  # [B, F, T]
        conv_mask = mask.transpose(1, 2)  # [B, 1, T]
        conv_input = conv_input * conv_mask
        
        encodings, masks = self.encoder(conv_input, conv_mask)
        decoder_output = self.decoder(list(reversed(encodings)), list(reversed(masks)))
        
        # Back to sequence format for LSTM (batch_first)
        decoder_output = decoder_output.transpose(1, 2)  # [B, T, F]
        
        # Post-processing with LSTM 
        packed_decoder = nn.utils.rnn.pack_padded_sequence(
            decoder_output, 
            lengths=seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_post_lstm, _ = self.post_lstm(packed_decoder)
        post_lstm, _ = nn.utils.rnn.pad_packed_sequence(packed_post_lstm, batch_first=True)
        
        post_forward = post_lstm[:, :, :self.post_lstm_units]
        post_backward = post_lstm[:, :, self.post_lstm_units:]
        post_lstm = post_forward * post_backward
        
        # Convert back to channel-first for final conv and apply mask
        conv_output = post_lstm.transpose(1, 2)  # [B, F, T]
        conv_output = conv_output * conv_mask
        
        logits = self.final_conv(conv_output)  # [B, 2, T]
        
        mask_bool = conv_mask.bool()
        preds = torch.zeros_like(logits)
        preds[:, 1, :] = 1.0  # Default: [0,1]
        
        # Apply softmax only at masked positions
        softmaxed = F.softmax(logits, dim=1)
        preds = torch.where(mask_bool, softmaxed, preds)
        
        return preds


if __name__ == "__main__":
    params = {
        'input_dim': 5,
        'pre_lstm_units': 64,
        'post_lstm_units': 64,
        'enc_filters': [80, 96, 48],
        'enc_kernels': [12, 10, 8],
        'dec_filters': [96, 112, 96],
        'dec_kernels': [10, 12, 14],
        'last_kernel': 4,
    }
    
    try:
        print("Initializing model...")
        model = UNetModel(**params)
        print(f"Model initialized: {model.__class__.__name__}")
        
        x = torch.randn(16, 100, 5)  # [B, T, F]
        mask = torch.ones(16, 100, 1)  # [B, T, 1]
        print(f"Input shapes - x: {x.shape}, mask: {mask.shape}")
        
        print("Starting forward pass...")
        y = model(x, mask)
        print(f"Forward pass successful! Output shape: {y.shape}")
        
        print("Model test completed successfully!")
    
    except Exception as e:
        import traceback
        print(f"Error during model test: {e}")
        traceback.print_exc()
