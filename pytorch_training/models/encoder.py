import torch
import torch.nn as nn
from .layers import GradientReversal


class BatchNorm1dTranspose(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerEncoderLayerBN(nn.TransformerEncoderLayer):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(d_model, *args, **kwargs)
        self.norm1 = BatchNorm1dTranspose(d_model)
        self.norm2 = BatchNorm1dTranspose(d_model)


class Encoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dim_feedforward_size,
        n_heads,
        out_size,
        dropout_p,
        use_batch_norm=False,
        second_head_out_size=None,
        use_cls_token=False,
        return_only_cls_token=False,
        return_hidden=False,
        return_hiddens_by_layers=False,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.first_layer = nn.Linear(in_features, hidden_size)
        if not use_batch_norm:
            enc_layer = nn.TransformerEncoderLayer(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        else:
            enc_layer = TransformerEncoderLayerBN(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size, bias=False)

        self.class_token = (
            nn.Parameter(
                torch.randn(1, 1, hidden_size),
                requires_grad=True,
            )
            if use_cls_token
            else None
        )
        self.return_only_cls_token = return_only_cls_token
        self.second_head = (
            nn.Linear(hidden_size, second_head_out_size)
            if second_head_out_size is not None
            else None
        )
        self.return_hidden = return_hidden
        self.return_hiddens_by_layers = return_hiddens_by_layers

    def forward(self, x, mask):

        mask = (~mask).float()
        x = self.first_layer(x)
        hiddens_by_layer = []

        if self.class_token is not None:
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            mask = torch.cat(
                [torch.ones(x.shape[0], 1, dtype=torch.float32).to(mask.device), mask],
                dim=1,
            )

        if self.return_hiddens_by_layers:
            for layer in self.enc.layers:
                x = layer(x, src_key_padding_mask=mask)
                hiddens_by_layer.append(x)
            y = self.head(x)
        else:
            x = self.enc(x, src_key_padding_mask=mask)
            y = self.head(x)

        if self.second_head is not None:
            z = self.second_head(x)
            res = torch.cat([y.mean(1), z.mean(1)], dim=-1)
        if self.class_token is not None and self.return_only_cls_token:
            res = y.mean(1)
        else:
            res = y

        if self.return_hiddens_by_layers:
            return res, hiddens_by_layer
        elif self.return_hidden:
            return res, x
        else:
            return res


class EncoderDomainAdaptation(nn.Module):
    def __init__(
        self,
        num_domains=2,
        domain_classifier_hidden_size=128,
        domain_classifier_layers=2,
        gradient_reversal_alpha=1.0,
        **kwargs
    ):
        super().__init__()

        self.encoder = Encoder(**kwargs)
        self.encoder.return_hidden = True
        self.angle_head = nn.Linear(self.encoder.hidden_size, self.encoder.out_size)
        self.gradient_reversal = GradientReversal(alpha=gradient_reversal_alpha)

        domain_classifier_layers_list = []
        input_size = self.encoder.hidden_size

        for _ in range(domain_classifier_layers - 1):
            domain_classifier_layers_list.extend(
                [
                    nn.Linear(input_size, domain_classifier_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.encoder.dropout_p),
                ]
            )
            input_size = domain_classifier_hidden_size

        domain_classifier_layers_list.append(nn.Linear(input_size, num_domains))

        self.domain_classifier = nn.Sequential(*domain_classifier_layers_list)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, mask):
        _, hidden_states = self.encoder(x, mask)
        features = hidden_states[:, 0]

        angle_output = self.angle_head(features)
        reversed_features = self.gradient_reversal(features)
        domain_output = self.domain_classifier(reversed_features)

        return angle_output, domain_output
