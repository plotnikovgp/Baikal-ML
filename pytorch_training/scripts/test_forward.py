import torch

from models import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_params = {
    "in_features": 5,
    "hidden_size": 512,
    "num_layers": 5,
    "dim_feedforward_size": 512,
    "n_heads": 1,
    "out_size": 2,
    "dropout_p": 0.0,
}

model = load_model("encoder", model_params).to(DEVICE)
model.eval()

batch_size = 4
seq_len = 100

x = torch.zeros(batch_size, seq_len, model_params["in_features"], device=DEVICE)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=DEVICE)

with torch.no_grad():
    output = model(x, mask)

print(f"Input shape: {x.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")
