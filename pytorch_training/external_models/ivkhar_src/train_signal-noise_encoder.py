import random as rd

import h5py as h5
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

num_h = 2
m_dim = 128 * 2
n_layers = 4
dr_rate_att = 0.0
dr_rate = 0.0
pred_units = [2]
activation = tf.keras.activations.gelu
# activation = tf.keras.layers.LeakyReLU(0.2)

num_par_calls_ds = 8

# predefine padding value
# (q,t,x,y,z, ma, lsx2, tres, ws)
dense_def_vals = np.zeros((10,), dtype=np.float32)
aux_idxs = [0, 1, 2, 3, 4, 7, 8, 9]
# WAS 4.
aux_vals = [-0.162, 3.0, 3.0, 3.0, 3.0, 1.0, 1e5, 1.0]
dense_def_vals[aux_idxs] = aux_vals
dense_def_vals = tf.constant(dense_def_vals)


# eliminate OMs with Q below threshold
def eliminate_low_Q(gen, data, labels):
    non_low_Q_mask = data[:, 0] > gen.Q_low_lim_norm
    n_data = data[non_low_Q_mask]
    labels = labels[non_low_Q_mask]
    return (n_data, n_labels)


# make labels, relabel if required
def make_2cl_labels(true_labels, relabel_big_tres, time_limit, weights):
    t_res = true_labels[:, 1]
    true_labels = true_labels[:, 0]
    # identify true signal idxs
    idxs_true_signal = np.nonzero(true_labels != 0)
    # make one hot labels
    labels_one_hot = np.full(np.concatenate((true_labels.shape, [2])), [0.0, 1.0], dtype=np.float32)
    labels_one_hot[idxs_true_signal] = [1.0, 0.0]
    # set weights
    weights_out = np.full(true_labels.shape, weights[1])
    weights_out[idxs_true_signal] = weights[0]
    # relabel big tres, if requested
    if relabel_big_tres:
        # identify signal hits with big t_res
        idxs_big_res = np.nonzero((np.abs(t_res) > time_limit) * (true_labels != 0))
        labels_one_hot[idxs_big_res] = [0.0, 1.0]
        weights_out[idxs_big_res] = weights[1]
    labels = np.concatenate((labels_one_hot, np.expand_dims(t_res, axis=-1)), axis=-1)
    return (labels, weights_out)


# addative noise
def add_gauss(data, labels, ws, g_add_stds, t_std, ev_starts):
    g_add_stds = np.broadcast_to(g_add_stds, data.shape)
    noise = np.random.normal(scale=g_add_stds, size=data.shape)
    data += noise
    # correct t_res
    labels[:, -1] = labels[:, -1] + noise[:, 1] * t_std
    sort_idxs = np.concatenate(
        [
            np.argsort(data[ev_starts[i] : ev_starts[i + 1], 1], axis=0) + ev_starts[i]
            for i in range(len(ev_starts) - 1)
        ]
    )
    data = data[sort_idxs]
    labels = labels[sort_idxs]
    ws = ws[sort_idxs]
    return (data, labels, ws)


# mult noise
class gauss_mult_noise:
    def __init__(self, Q_mean_noise, n_fraction):
        self.Q_mean_noise = Q_mean_noise
        self.n_fraction = n_fraction

    def make_noise(self, Qs):
        noises = np.random.normal(scale=self.n_fraction, size=Qs.shape)
        Qs = Qs + (Qs + self.Q_mean_noise)
        return Qs


# generator without shuffling
class generator_no_shuffle:
    def __init__(
        self,
        file,
        regime,
        batch_size,
        return_reminder,
        set_up_Q_lim,
        up_Q_lim,
        set_low_Q_lim,
        low_Q_lim,
        relabel_big_tres,
        time_limit,
        weights,
        apply_add_gauss,
        g_add_stds,
        apply_mult_gauss,
        q_noise_fraction,
    ):
        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        self.return_reminder = return_reminder
        self.set_up_Q_lim = set_up_Q_lim
        self.set_low_Q_lim = set_low_Q_lim
        self.time_limit = time_limit
        self.relabel_big_tres = relabel_big_tres
        self.weights = weights.astype(np.float32)
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_add_stds = g_add_stds
        self.hf = h5.File(self.file, "r")

        self.num = self.hf[self.regime + "/ev_starts/data"].shape[0] - 1
        self.means = self.hf["norm_param/mean"][()]
        self.stds = self.hf["norm_param/std"][()]
        self.Q_low_lim_norm = (low_Q_lim - self.means[0]) / self.stds[0]
        if set_up_Q_lim:
            Q_mean = self.hf["norm_param/mean"][0]
            Q_std = self.hf["norm_param/std"][0]
            self.Q_up_lim_norm = (up_Q_lim - Q_mean) / Q_std
        if apply_mult_gauss:
            Q_mean = self.hf["norm_param/mean"][0]
            Q_std = self.hf["norm_param/std"][0]
            self.mult_gauss = gauss_mult_noise(Q_mean / Q_std, q_noise_fraction)
        self.batch_num = self.num // self.batch_size
        if return_reminder and (self.num % self.batch_size) != 0:
            self.batch_num += 1

    def step(self, start, stop, loc_ev_idxs):
        data = self.hf[self.regime + "/data/data"][start:stop]
        true_labels = np.expand_dims(
            self.hf[self.regime + "/labels/data"][start:stop].astype(np.float32), axis=-1
        )
        t_res = self.hf[self.regime + "/t_res/data"][start:stop]
        true_labels = np.concatenate((true_labels, np.expand_dims(t_res, axis=-1)), axis=-1)
        if self.set_up_Q_lim:
            data[:, 0] = np.where(data[:, 0] > self.Q_up_lim_norm, self.Q_up_lim_norm, data[:, 0])
        if self.set_low_Q_lim:
            (data, true_labels) = eliminate_low_Q(self, data, true_labels)
        (labels, ws) = make_2cl_labels(
            true_labels, self.relabel_big_tres, self.time_limit, self.weights
        )
        # apply noise
        if self.apply_add_gauss:
            (data, labels, ws) = add_gauss(
                data, labels, ws, self.g_add_stds, self.stds[1], loc_ev_idxs
            )
        if self.apply_mult_gauss:
            data[:, 0] = self.mult_gauss.make_noise(data[:, 0])
        return (data, labels, ws)

    def __call__(self):
        start = 0
        stop = self.batch_size
        for i in range(self.batch_num):
            ev_idxs = self.hf[self.regime + "/ev_starts/data"][start : stop + 1]
            loc_ev_idxs = ev_idxs - ev_idxs[0]
            out_data = self.step(ev_idxs[0], ev_idxs[-1], loc_ev_idxs)
            yield out_data + (np.diff(ev_idxs),)
            start += self.batch_size
            stop += self.batch_size


@tf.function
def flat_to_dense(data, labels, ws, raw_lens):
    # (data, labels, ws, raw_lens) = data_in
    # raw_lens = tf.cast(raw_lens, tf.int32)
    mask = tf.fill(tf.shape(data)[0:1], tf.cast(1.0, tf.float32))
    data = tf.concat(
        (data, tf.expand_dims(mask, axis=-1), labels, tf.expand_dims(ws, axis=-1)), axis=1
    )
    ragged = tf.RaggedTensor.from_row_lengths(data, raw_lens)
    dense = ragged.to_tensor(default_value=dense_def_vals)
    return (dense[:, :, :6], dense[:, :, 6:9], dense[:, :, 9])


### due to technical reason (tf side), bs is constant and no reminder
# redudant variables for compatibility with dense trainsing
def make_datasets(
    h5f,
    make_generator_shuffle,
    return_batch_reminder,
    train_batch_size,
    train_buffer_size,
    test_batch_size,
    max_len,
    set_up_Q_lim,
    up_Q_lim,
    set_low_Q_lim,
    low_Q_lim,
    relabel_big_tres,
    time_limit,
    weights,
    apply_add_gauss,
    g_add_stds,
    apply_mult_gauss,
    q_noise_fraction,
):
    # generator for training data
    tr_generator = generator_no_shuffle(
        h5f,
        "train",
        train_batch_size,
        return_batch_reminder,
        set_up_Q_lim,
        up_Q_lim,
        set_low_Q_lim,
        low_Q_lim,
        relabel_big_tres,
        time_limit,
        weights,
        apply_add_gauss,
        g_add_stds,
        apply_mult_gauss,
        q_noise_fraction,
    )
    if return_batch_reminder:
        # size of the last batch is unknown
        tr_batch_size = None
    else:
        tr_batch_size = train_batch_size

    train_dataset = tf.data.Dataset.from_generator(
        tr_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(tr_batch_size,), dtype=tf.int32),
        ),
    )

    train_dataset = (
        train_dataset.map(flat_to_dense, num_parallel_calls=num_par_calls_ds)
        .repeat(-1)
        .prefetch(num_par_calls_ds)
    )

    # generator for test data
    test_batch_size = tr_batch_size
    te_generator = generator_no_shuffle(
        h5f,
        "test",
        test_batch_size,
        False,
        set_up_Q_lim,
        up_Q_lim,
        set_low_Q_lim,
        low_Q_lim,
        relabel_big_tres,
        time_limit,
        weights,
        False,
        None,
        False,
        None,
    )

    test_dataset = tf.data.Dataset.from_generator(
        te_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(tr_batch_size,), dtype=tf.int32),
        ),
    )

    test_dataset = test_dataset.map(flat_to_dense).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def make_val_dataset(
    h5f,
    val_batch_size,
    max_len,
    set_up_Q_lim,
    up_Q_lim,
    set_low_Q_lim,
    low_Q_lim,
    relabel_big_tres,
    time_limit,
    weights,
):
    val_generator = generator_no_shuffle(
        h5f,
        "val",
        val_batch_size,
        False,
        set_up_Q_lim,
        up_Q_lim,
        set_low_Q_lim,
        low_Q_lim,
        relabel_big_tres,
        time_limit,
        weights,
        False,
        None,
        False,
        None,
    )

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(val_batch_size,), dtype=tf.int32),
        ),
    )

    val_dataset = val_dataset.map(flat_to_dense, num_parallel_calls=num_par_calls_ds).prefetch(
        num_par_calls_ds
    )
    return val_dataset


# entropy loss
class entropy_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss_name = "_entropy"

    def __call__(self, y_true, y_pred, sample_weight):
        label = y_true[:, :, :2]
        t_res = y_true[:, :, 2]
        entropy = tf.keras.losses.binary_crossentropy(label, y_pred)
        entropy = tf.math.multiply(entropy, sample_weight)
        loss = tf.math.reduce_mean(entropy)
        return loss


class accuracy_extr(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(accuracy_extr, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="acc", initializer="zeros")
        self.steps = self.add_weight(name="st", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_labels = y_true[:, :, :2]
        class_preds = tf.math.argmax(y_pred, axis=-1)
        class_true = tf.math.argmax(true_labels, axis=-1)
        mask = tf.where(y_true[:, :, 2] >= 9 * 1e4, False, True)
        correct_preds = tf.cast(tf.logical_and((class_preds == class_true), mask), self.dtype)
        correct_preds = tf.math.reduce_sum(correct_preds) / tf.reduce_sum(tf.cast(mask, tf.float32))
        self.accuracy.assign_add(correct_preds)
        self.steps.assign_add(1)

    def reset_state(self):
        self.accuracy.assign(0.0)
        self.steps.assign(0.0)

    def result(self):
        return self.accuracy / self.steps


class tres_extr(tf.keras.metrics.Metric):
    def __init__(self, t_lim, th, name="tres", **kwargs):
        super(tres_extr, self).__init__(name=name, **kwargs)
        self.tres = self.add_weight(name="tres", initializer="zeros")
        self.steps = self.add_weight(name="stt", initializer="zeros")
        self.t_lim = t_lim
        self.th = th

    def update_state(self, y_true, y_pred, sample_weight=None):
        signal_mask = tf.where(y_pred[:, :, 0] > self.th, 1.0, 0.0)
        sig_tres = tf.where(signal_mask == 1.0, tf.math.abs(y_true[:, :, 2]), 0.0)
        num_sigs = tf.reduce_sum(signal_mask)
        int_tres = tf.reduce_sum(sig_tres + 1e-9)
        self.tres.assign_add(num_sigs / int_tres)
        self.steps.assign_add(1)

    def reset_state(self):
        self.tres.assign(0.0)
        self.steps.assign(0.0)

    def result(self):
        return self.tres / self.steps


# projecting to Qs, Ks, Vs
class qkv_projector(tf.keras.layers.Layer):
    def __init__(self, qk_dim, v_dim):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim

    def build(self, input_shape):
        num_fs = input_shape[-1]
        self.proj_matrix_Q = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(num_fs, self.qk_dim)),
            trainable=True,
        )
        self.proj_matrix_K = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(num_fs, self.qk_dim)),
            trainable=True,
        )
        self.proj_matrix_V = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(num_fs, self.v_dim)),
            trainable=True,
        )
        self.bias_Q = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(self.qk_dim,)),
            trainable=True,
        )
        self.bias_K = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(self.qk_dim,)),
            trainable=True,
        )
        self.bias_V = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(self.v_dim,)), trainable=True
        )

    def call(self, node, training=False):
        qs = tf.linalg.matmul(node, self.proj_matrix_Q) + self.bias_Q
        ks = tf.linalg.matmul(node, self.proj_matrix_K) + self.bias_K
        vs = tf.linalg.matmul(node, self.proj_matrix_V) + self.bias_V
        return (qs, ks, vs)


# attention calculation
class NLPAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.norm_softmax = 1.0 / tf.math.sqrt(tf.cast(input_shape[1][-1], tf.float32))

    def call(self, inputs, training=False):
        qs, ks, vs = inputs
        prods = tf.linalg.matmul(qs, ks, transpose_b=True)
        att_scores = tf.nn.softmax(prods * self.norm_softmax)
        messgs = tf.linalg.matmul(att_scores, vs)
        return messgs


# multihead
class multiheadAttentionNLP(tf.keras.layers.Layer):
    def __init__(self, num_heads, qk_dim, v_dim, out_dim, dr_rate):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.prog_layer = qkv_projector(num_heads * qk_dim, num_heads * v_dim)
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.att_layer = NLPAttention()
        self.dropout = tf.keras.layers.Dropout(dr_rate)

    def build(self, input_shape):
        qk_shape = input_shape[:-1] + (self.qk_dim,)
        v_shape = input_shape[:-1] + (self.v_dim,)
        self.att_layer.build((qk_shape, qk_shape, v_shape))
        self.matrix_out = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(
                shape=(self.num_heads * self.v_dim, self.out_dim)
            ),
            trainable=True,
        )
        self.prog_layer.build(input_shape)
        self.bias = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=(self.out_dim,)),
            trainable=True,
        )

    def call(self, nodes, training=False):
        x = self.dropout(nodes)
        (qs, ks, vs) = self.prog_layer(x)
        qs = tf.stack(tf.split(qs, self.num_heads, axis=-1), axis=0)
        ks = tf.stack(tf.split(ks, self.num_heads, axis=-1), axis=0)
        vs = tf.stack(tf.split(vs, self.num_heads, axis=-1), axis=0)
        msgs = self.att_layer((qs, ks, vs))
        msgs = tf.concat(tf.unstack(msgs, axis=0), axis=-1)
        res = tf.linalg.matmul(msgs, self.matrix_out) + self.bias
        return res


class dense_block(tf.keras.layers.Layer):
    def __init__(self, unit, dr_rate, use_bn, activation):
        super().__init__()
        self.dense = tf.keras.layers.Dense(unit)
        self.dropout = tf.keras.layers.Dropout(dr_rate)
        self.activation = activation
        if use_bn:
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.norm_layer = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.dropout(inputs, training=training)
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm_layer(x, training=training)
        return x


# updating layer
class predict_layer(tf.keras.layers.Layer):
    def __init__(self, units, dr_rate, use_bn, activation):
        super().__init__()
        self.d_layers = []
        self.num_ls = len(units)
        for un in units:
            self.d_layers.append(dense_block(un, dr_rate, use_bn, activation))
            # self.bn_layers.append( tf.keras.layers.BatchNormalization() )

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_ls):
            x = self.d_layers[i](x, training=training)
            # x = self.bn_layers[i](x, training=training)
        preds = tf.nn.softmax(x)
        return preds


class update_layer(tf.keras.layers.Layer):
    def __init__(self, units, dr_rate, use_bn, activation):
        super().__init__()
        self.layers = []
        for un in units:
            self.layers.append(dense_block(un, dr_rate, use_bn, activation))
        self.add = tf.keras.layers.Add()
        if use_bn:
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.norm_layer = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        # concat
        updt = tf.concat((x, inputs), axis=-1)
        # add
        # updt = self.add([x,inputs])
        # updt = self.norm_layer(updt)
        return updt


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_layer, updater_layer, use_bn):
        super().__init__()
        self.attention_layer = attention_layer
        self.upd_layer = updater_layer
        if use_bn:
            self.layernorm = tf.keras.layers.BatchNormalization()
        else:
            self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False):
        data, mask = inputs
        messgs = self.attention_layer(data, training=training)

        # var 1: add, remove self.add if not needed
        # updts = self.add( [data,messgs] )
        # var 2: concat
        updts = tf.concat((data, messgs), axis=-1)

        updts = self.layernorm(updts, training=training)
        updated = self.upd_layer(updts, training=training) * mask
        return updated


class Encoder(tf.keras.Model):
    def __init__(self, enc_layers):
        super().__init__()
        self.enc_layers = enc_layers
        self.depth = len(enc_layers)

    def call(self, inputs, training=False):
        mask = inputs[:, :, -1:]
        x = inputs
        for i in range(self.depth):
            x = self.enc_layers[i]((x, mask), training=training)
        preds = tf.where(tf.cast(mask, bool), x, tf.constant([0.0, 1.0]))
        return preds


max_len = None

num_heads = [num_h for _ in range(n_layers)]
qk_dims = [m_dim for _ in range(n_layers)]
v_dims = [m_dim for _ in range(n_layers)]
out_dims = [m_dim for _ in range(n_layers)]

# gauss noise
# 0.1 ~ 1 p.e., 150 ns, 4, 4, 15 m
apply_add_gauss = False
# stds_gauss = [0.03, 0.005, 0.005, 0.005, 0.0003]
stds_gauss = [0.02, 0.001, 0.001, 0.001, 0.0002]
apply_mult_gauss = False
Q_noise_fraction = 0.1

# limiting Q vals
set_up_Q_lim = True
up_Q_lim = 100
set_low_Q_lim = False
low_Q_lim = 0.0

# coefficient for t_res penalty
t_res_pen_coeff = 0.1  # old mc
t_res_pen_coeff = 0.01  # new mc dense
t_res_pen_coeff = 0.3  # new mc flat
loss_norm_coeff = 0.75
time_limit = 20.0
# whether mark signal hits with big t_res as noise
relabel_big_tres = False
# for focal loss
gamma = 2

# weights for signal/noise
weights = np.array([1.0, 1.0])
avg_w = np.sum(weights) / 2

h5f = "/home2/ivkhar/Baikal/data/baikal_muatm-multi_0523_flat_norm.h5"

bs = 64
val_batch_size = bs
learning_rate = 1e-5 * 4
# learning_rate=0.0003
# learning_rate = 0.0016*0.3

train_dataset, test_dataset = make_datasets(
    h5f,
    True,
    False,
    bs,
    1250 * 64,
    64,
    max_len,
    set_up_Q_lim,
    up_Q_lim,
    set_low_Q_lim,
    low_Q_lim,
    relabel_big_tres,
    time_limit,
    weights,
    apply_add_gauss,
    stds_gauss,
    apply_mult_gauss,
    Q_noise_fraction,
)

val_dataset = make_val_dataset(
    h5f,
    val_batch_size,
    max_len,
    set_up_Q_lim,
    up_Q_lim,
    set_low_Q_lim,
    low_Q_lim,
    relabel_big_tres,
    time_limit,
    weights,
)

att_layers = [
    multiheadAttentionNLP(num_heads=nh, qk_dim=qd, v_dim=vd, out_dim=od, dr_rate=dr_rate_att)
    for (nh, qd, vd, od) in zip(num_heads, qk_dims, v_dims, out_dims)
]

upd_units = [[2 * m_dim, m_dim] for _ in range(n_layers - 1)]
upd_layers = [update_layer(units, dr_rate_att, True, activation) for units in upd_units] + [
    predict_layer(pred_units, dr_rate, True, activation)
]

assert len(upd_layers) == len(att_layers)

enc_layers = [EncoderLayer(att, upd, True) for (att, upd) in zip(att_layers, upd_layers)]

encoder = Encoder(enc_layers)

pre_layer = dense_block(m_dim, 0.1, True, activation)

loss_fn = entropy_loss()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
encoder.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[accuracy_extr(), tres_extr(20.0, 0.7)],
    weighted_metrics=[],
)

prefix_save = "/home/ivkhar/Baikal/models/"
model_name = "encoder_no-relab_no-noise_entropy_" + "bs-" + str(bs) + "_lr-" + str(learning_rate)
model_name += (
    "_num-hs-"
    + str(num_h)
    + "_m-dim-"
    + str(m_dim)
    + "_num-ls-"
    + str(n_layers)
    + "_drop-"
    + str(dr_rate_att)
    + "_n-h-"
    + str(num_h)
)
model_name += (
    "_background_w-"
    + "-".join([str(w) for w in weights])
    + "_stds_g-"
    + "-".join([str(stds) for stds in stds_gauss])
)
model_name += (
    "_Qup-"
    + str(set_up_Q_lim)[0]
    + "-"
    + str(up_Q_lim)
    + "_Qlow-"
    + str(set_low_Q_lim)[0]
    + "-"
    + str(low_Q_lim)
)
model_name += "_relab-" + str(relabel_big_tres)[0] + "-" + str(time_limit)
save_path = prefix_save + model_name

EarlyStopCall = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
)
CheckPointCall = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_path + "_ckpt",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    save_freq="epoch",
)
LR_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.25,
    patience=3,
)
TB_cb = tf.keras.callbacks.TensorBoard(
    log_dir="/home/ivkhar/Baikal/fit_logs/" + model_name, update_freq="epoch"
)
callbacks = [EarlyStopCall, CheckPointCall, LR_cb, TB_cb]

encoder.fit(
    train_dataset,
    steps_per_epoch=5000,
    validation_steps=1000,
    epochs=250,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=0,
)
encoder.save(save_path)
ev_res = encoder.evaluate(val_dataset, return_dict=True, verbose=2)
with open(save_path + ".txt", "a") as f:
    keys = ev_res.keys()
    messege = [key + ": " + str(ev_res[key]) for key in keys]
    f.write("Model " + str(j + start_idxs) + " ; ")
    f.write(", ".join(messege) + "\n")
