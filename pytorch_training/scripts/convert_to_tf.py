import numpy as np
import tensorflow as tf
import torch

from models import load_model


class TFTransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=True)

        self.ffn1 = tf.keras.layers.Dense(dim_feedforward, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(d_model)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        head_dim = self.d_model // self.n_heads
        q = tf.reshape(q, [batch_size, seq_len, self.n_heads, head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.n_heads, head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.n_heads, head_dim])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attn_weights = tf.matmul(q, k, transpose_b=True) / scale

        if mask is not None:
            attn_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
            attn_mask = (1.0 - attn_mask) * -1e9
            attn_weights = attn_weights + attn_mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights, v)

        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.d_model])

        attn_output = self.wo(attn_output)
        attn_output = self.dropout1(attn_output, training=training)

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn2(self.ffn1(x))
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.norm2(x + ffn_output)


class TFEncoder(tf.keras.Model):
    def __init__(
        self, in_features, hidden_size, num_layers, dim_feedforward, n_heads, out_size, dropout=0.0
    ):
        super().__init__()
        self.first_layer = tf.keras.layers.Dense(hidden_size)
        self.encoder_layers = [
            TFTransformerEncoderLayer(
                hidden_size, n_heads, dim_feedforward, dropout, name=f"layer_{i}"
            )
            for i in range(num_layers)
        ]
        self.head = tf.keras.layers.Dense(out_size, use_bias=False)

    def call(self, x, mask=None, training=False):
        x = self.first_layer(x)
        for layer in self.encoder_layers:
            x = layer(x, mask=mask, training=training)
        return self.head(x)


def convert_pytorch_to_tf(checkpoint_path, model_params, output_path):
    pt_model = load_model("encoder", model_params)
    pt_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    pt_model.eval()

    tf_model = TFEncoder(
        in_features=model_params["in_features"],
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_layers"],
        dim_feedforward=model_params["dim_feedforward_size"],
        n_heads=model_params["n_heads"],
        out_size=model_params["out_size"],
        dropout=model_params["dropout_p"],
    )

    dummy_input = tf.zeros((1, 10, model_params["in_features"]))
    dummy_mask = tf.ones((1, 10), dtype=tf.bool)
    _ = tf_model(dummy_input, dummy_mask)

    pt_state = pt_model.state_dict()

    tf_model.first_layer.set_weights(
        [
            pt_state["first_layer.weight"].numpy().T,
            pt_state["first_layer.bias"].numpy(),
        ]
    )

    for i, tf_layer in enumerate(tf_model.encoder_layers):
        prefix = f"enc.layers.{i}"

        in_proj_weight = pt_state[f"{prefix}.self_attn.in_proj_weight"].numpy()
        in_proj_bias = pt_state[f"{prefix}.self_attn.in_proj_bias"].numpy()

        wq, wk, wv = np.split(in_proj_weight, 3, axis=0)
        bq, bk, bv = np.split(in_proj_bias, 3, axis=0)

        tf_layer.wq.set_weights([wq.T, bq])
        tf_layer.wk.set_weights([wk.T, bk])
        tf_layer.wv.set_weights([wv.T, bv])

        tf_layer.wo.set_weights(
            [
                pt_state[f"{prefix}.self_attn.out_proj.weight"].numpy().T,
                pt_state[f"{prefix}.self_attn.out_proj.bias"].numpy(),
            ]
        )

        tf_layer.ffn1.set_weights(
            [
                pt_state[f"{prefix}.linear1.weight"].numpy().T,
                pt_state[f"{prefix}.linear1.bias"].numpy(),
            ]
        )
        tf_layer.ffn2.set_weights(
            [
                pt_state[f"{prefix}.linear2.weight"].numpy().T,
                pt_state[f"{prefix}.linear2.bias"].numpy(),
            ]
        )

        tf_layer.norm1.set_weights(
            [
                pt_state[f"{prefix}.norm1.weight"].numpy(),
                pt_state[f"{prefix}.norm1.bias"].numpy(),
            ]
        )
        tf_layer.norm2.set_weights(
            [
                pt_state[f"{prefix}.norm2.weight"].numpy(),
                pt_state[f"{prefix}.norm2.bias"].numpy(),
            ]
        )

    tf_model.head.set_weights(
        [
            pt_state["head.weight"].numpy().T,
        ]
    )

    tf_model.save(output_path)
    print(f"TensorFlow model saved to {output_path}")

    return pt_model, tf_model


def verify_conversion(pt_model, tf_model, model_params):
    np.random.seed(42)
    batch_size, seq_len = 2, 50
    x_np = np.random.randn(batch_size, seq_len, model_params["in_features"]).astype(np.float32)
    mask_np = np.ones((batch_size, seq_len), dtype=bool)

    x_pt = torch.tensor(x_np)
    mask_pt = torch.tensor(mask_np)

    with torch.no_grad():
        pt_out = pt_model(x_pt, mask_pt).numpy()

    x_tf = tf.constant(x_np)
    mask_tf = tf.constant(mask_np)
    tf_out = tf_model(x_tf, mask_tf, training=False).numpy()

    diff = np.abs(pt_out - tf_out).max()
    mean_diff = np.abs(pt_out - tf_out).mean()
    print(f"Max difference: {diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"PyTorch output shape: {pt_out.shape}")
    print(f"TensorFlow output shape: {tf_out.shape}")
    print(f"PyTorch sample: {pt_out[0, 0, :]}")
    print(f"TensorFlow sample: {tf_out[0, 0, :]}")

    if diff < 1e-4:
        print("Conversion successful!")
    else:
        print("Warning: outputs differ")


if __name__ == "__main__":
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
    output_path = "./tf_encoder_model"

    pt_model, tf_model = convert_pytorch_to_tf(checkpoint_path, model_params, output_path)
    verify_conversion(pt_model, tf_model, model_params)
