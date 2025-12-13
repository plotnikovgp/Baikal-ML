import numpy as np
import torch

from models import load_model

model_params = {
    "in_features": 5,
    "hidden_size": 512,
    "num_layers": 5,
    "dim_feedforward_size": 512,
    "n_heads": 1,
    "out_size": 2,
    "dropout_p": 0.0,
}

checkpoint_path = "/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt"

pt_model = load_model("encoder", model_params)
pt_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
pt_model.eval()

pt_state = pt_model.state_dict()

print("PyTorch state dict keys:")
for k, v in pt_state.items():
    print(f"  {k}: {v.shape}")

# Test with simple input
x_np = np.random.randn(1, 10, 5).astype(np.float32)
mask_np = np.ones((1, 10), dtype=bool)

x_pt = torch.tensor(x_np)
mask_pt = torch.tensor(mask_np)

# Check first layer output
with torch.no_grad():
    first_out_pt = pt_model.first_layer(x_pt)
    print(
        f"\nPyTorch first_layer output: mean={first_out_pt.mean():.6f}, std={first_out_pt.std():.6f}"
    )

    # Check mask transformation
    mask_transformed = (~mask_pt).float()
    print(f"PyTorch mask (transformed for attention): {mask_transformed}")

    # Run through encoder
    enc_out = pt_model.enc(first_out_pt, src_key_padding_mask=mask_transformed)
    print(f"PyTorch encoder output: mean={enc_out.mean():.6f}, std={enc_out.std():.6f}")

    final_out = pt_model.head(enc_out)
    print(f"PyTorch final output: mean={final_out.mean():.6f}, std={final_out.std():.6f}")
    print(f"PyTorch final output sample: {final_out[0, 0, :]}")
