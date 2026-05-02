# import torch
# import torch.nn as nn


# class ISeeCube(GNN):
#     """ISeeCube model."""

#     def __init__(
#         self,
#         hidden_dim: int = 384,
#         seq_length: int = 196,
#         num_layers: int = 16,
#         num_heads: int = 12,
#         mlp_dim: int = 1536,
#         rel_pos_buckets: int = 32,
#         max_rel_pos: int = 256,
#         num_register_tokens: int = 3,
#         scaled_emb: bool = False,
#         n_features: int = 6,
#     ):
#         """Construct `ISeeCube`.

#         Args:
#             hidden_dim: The latent feature dimension.
#             seq_length: The number of pulses in a neutrino event.
#             num_layers: The depth of the transformer.
#             num_heads: The number of the attention heads.
#             mlp_dim: The mlp dimension of FourierEncoder and Transformer.
#             rel_pos_buckets: Relative position buckets for relative position
#                 bias.
#             max_rel_pos: Maximum relative position for relative position bias.
#             num_register_tokens: The number of register tokens.
#             scaled_emb: Whether to scale the sinusoidal positional embeddings.
#             n_features: The number of features in the input data.
#         """
#         super().__init__(seq_length, hidden_dim)
#         # self.fourier_ext = FourierEncoder(
#         #     seq_length=seq_length,
#         #     mlp_dim=mlp_dim,
#         #     output_dim=hidden_dim,
#         #     scaled=scaled_emb,
#         #     n_features=n_features,
#         # )
#         # self.pos_embedding = nn.Parameter(
#         #     torch.empty(1, seq_length, hidden_dim).normal_(std=0.02),
#         #     requires_grad=True,
#         # )

#         self.class_token = nn.Parameter(
#             torch.empty(1, 1, hidden_dim),
#             requires_grad=True,
#         )
#         self.register_tokens = nn.Parameter(
#             torch.empty(1, num_register_tokens, hidden_dim),
#             requires_grad=True,
#         )
#     nn.TransformerEncoderLayer(
#                 hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
#             )
#         encoder_config = EncoderConfig(
#             encoder_attention_heads=num_heads,
#             encoder_embed_dim=hidden_dim,
#             encoder_ffn_embed_dim=mlp_dim,
#             encoder_layers=num_layers,
#             rel_pos_buckets=rel_pos_buckets,
#             max_rel_pos=max_rel_pos,
#         )
#         self.encoder = Encoder(encoder_config)

#         self.layer_norm = nn.LayerNorm(hidden_dim)

#     def forward(self, data: Data) -> Tensor:
#         """Apply learnable forward pass."""
#         x, _, _ = array_to_sequence(data.x, data.batch, padding_value=0)
#         x = self.fourier_ext(x)
#         batch_size = x.shape[0]

#         x += self.pos_embedding

#         batch_class_token = self.class_token.expand(batch_size, -1, -1)
#         batch_register_tokens = self.register_tokens.expand(batch_size, -1, -1)
#         x = torch.cat([batch_class_token, batch_register_tokens, x], dim=1)
