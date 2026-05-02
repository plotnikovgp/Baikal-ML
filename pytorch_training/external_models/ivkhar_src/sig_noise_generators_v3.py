# addative and mult gaussian noise inside generators

import random as rd

import h5py as h5
import numpy as np
import tensorflow as tf


# eliminate OMs with Q below threshold
def eliminate_low_Q(gen, data, labels):
    non_low_Q_idxs = np.nonzero(data[:, :, 0] > gen.Q_low_lim_norm)
    non_low_Q_mask = data[:, :, 0] > gen.Q_low_lim_norm
    num_oms_pass = np.sum(non_low_Q_mask, axis=-1)
    wr_mask = np.full(data.shape[:2], True)
    for i in range(data.shape[0]):
        wr_mask[i, num_oms_pass[i] :] = False
    n_data = np.full(data.shape, -gen.means / gen.stds)
    n_data[wr_mask] = data[non_low_Q_idxs]
    # labels as true flags; last dim is t_res
    n_labels = np.full(labels.shape, [0.0, 1e5])
    n_labels[wr_mask] = labels[non_low_Q_idxs]
    # make proper aux vals
    wr_mask = np.expand_dims(wr_mask, axis=-1)
    n_data[:, :, gen.aux_idxs] = np.where(wr_mask, n_data[:, :, gen.aux_idxs], gen.aux_vals)
    return (n_data, n_labels, wr_mask)


# make labels, relabel if required
def make_2cl_labels(true_labels, relabel_big_tres, time_limit, weights):
    t_res = true_labels[..., 1]
    true_labels = true_labels[..., 0]
    # identify true signal idxs
    idxs_true_signal = np.nonzero(true_labels != 0)
    # make one hot labels
    labels_one_hot = np.full(np.concatenate((true_labels.shape, [2])), [0.0, 1.0])
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
def add_gauss(data_in, labels, ws, g_add_stds):
    data = data_in[:, :, :-1]
    mask = data_in[:, :, -1:]
    g_add_stds = np.broadcast_to(g_add_stds, data.shape)
    noise = np.random.normal(scale=g_add_stds, size=data.shape) * mask
    data += noise
    sort_idxs = np.argsort(data[:, :, 1:2], axis=1)
    data = np.take_along_axis(data, sort_idxs, axis=1)
    data = np.concatenate((data, mask), axis=-1)
    labels = np.take_along_axis(labels, sort_idxs, axis=1)
    ws = np.squeeze(np.take_along_axis(np.expand_dims(ws, axis=-1), sort_idxs, axis=1), axis=-1)
    return (data, labels, ws)


# mult noise
class gauss_mult_noise:
    def __init__(self, Q_mean_noise, n_fraction):
        self.Q_mean_noise = Q_mean_noise
        self.n_fraction = n_fraction

    def make_noise(self, Qs, mask):
        noises = np.random.normal(scale=self.n_fraction, size=Qs.shape) * mask
        Qs = Qs + (Qs + self.Q_mean_noise) * noises
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
        self.weights = weights
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_add_stds = g_add_stds
        self.hf = h5.File(self.file, "r")

        self.num = self.hf[self.regime + "/data/data"].shape[0]
        if set_up_Q_lim:
            Q_mean = self.hf["norm_param/mean"][0]
            Q_std = self.hf["norm_param/std"][0]
            self.Q_up_lim_norm = (up_Q_lim - Q_mean) / Q_std
        if set_low_Q_lim:
            self.means = self.hf["norm_param/mean"][()]
            self.stds = self.hf["norm_param/std"][()]
            self.Q_low_lim_norm = (low_Q_lim - self.means[0]) / self.stds[0]
            self.aux_idxs = self.hf["aux_mask/idxs"][()]
            self.aux_vals = self.hf["aux_mask/vals"][()]
        if apply_mult_gauss:
            Q_mean = self.hf["norm_param/mean"][0]
            Q_std = self.hf["norm_param/std"][0]
            self.mult_gauss = gauss_mult_noise(Q_mean / Q_std, q_noise_fraction)
        self.batch_num = self.num // self.batch_size
        if return_reminder and (self.num % self.batch_size) != 0:
            self.batch_num += 1

    def step(self, start, stop):
        data = self.hf[self.regime + "/data/data"][start:stop]
        mask_channel = np.expand_dims(self.hf[self.regime + "/mask/data"][start:stop], axis=-1)
        true_labels = np.expand_dims(self.hf[self.regime + "/labels/data"][start:stop], axis=-1)
        t_res = self.hf[self.regime + "/t_res/data"][start:stop]
        true_labels = np.concatenate((true_labels, np.expand_dims(t_res, axis=-1)), axis=-1)
        if self.set_up_Q_lim:
            data[:, :, 0] = np.where(
                data[:, :, 0] > self.Q_up_lim_norm, self.Q_up_lim_norm, data[:, :, 0]
            )
        if self.set_low_Q_lim:
            (data, true_labels, mask_channel) = eliminate_low_Q(self, data, true_labels)
        data = np.concatenate((data, mask_channel), axis=-1)
        (labels, ws) = make_2cl_labels(
            true_labels, self.relabel_big_tres, self.time_limit, self.weights
        )
        # apply noise
        if self.apply_add_gauss:
            (data, labels, ws) = add_gauss(data, labels, ws, self.g_add_stds)
        if self.apply_mult_gauss:
            data[:, :, 0] = self.mult_gauss.make_noise(data[:, :, 0], data[:, :, -1])
        return (data, labels, ws)

    def __call__(self):
        start = 0
        stop = self.batch_size
        for i in range(self.batch_num):
            out_data = self.step(start, stop)
            yield out_data
            start += self.batch_size
            stop += self.batch_size


# generator with shuffling
class generator_with_shuffle:
    def __init__(
        self,
        file,
        regime,
        batch_size,
        buffer_size,
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
        self.buffer_size = buffer_size
        self.set_up_Q_lim = set_up_Q_lim
        self.set_low_Q_lim = set_low_Q_lim
        self.time_limit = time_limit
        self.relabel_big_tres = relabel_big_tres
        self.weights = weights
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_add_stds = g_add_stds
        self.hf = h5.File(self.file, "r")

        with h5.File(self.file, "r") as hf:
            self.num = self.hf[self.regime + "/data/data"].shape[0]
            if set_up_Q_lim:
                Q_mean = self.hf["norm_param/mean"][0]
                Q_std = self.hf["norm_param/std"][0]
                self.Q_up_lim_norm = (up_Q_lim - Q_mean) / Q_std
            if set_low_Q_lim:
                self.means = self.hf["norm_param/mean"][()]
                self.stds = self.hf["norm_param/std"][()]
                self.Q_low_lim_norm = (low_Q_lim - self.means[0]) / self.stds[0]
                self.aux_idxs = self.hf["aux_mask/idxs"][()]
                self.aux_vals = self.hf["aux_mask/vals"][()]
            if apply_mult_gauss:
                Q_mean = self.hf["norm_param/mean"][0]
                Q_std = self.hf["norm_param/std"][0]
                self.mult_gauss = gauss_mult_noise(Q_mean / Q_std, q_noise_fraction)
        self.batch_num = (self.num - self.buffer_size) // self.batch_size
        self.last_batches_num = self.buffer_size // self.batch_size
        if return_reminder and (self.num % self.batch_size) != 0:
            self.last_batches_num += 1

    def step(self, start, stop):
        data = self.hf[self.regime + "/data/data"][start:stop]
        mask_channel = np.expand_dims(self.hf[self.regime + "/mask/data"][start:stop], axis=-1)
        true_labels = np.expand_dims(self.hf[self.regime + "/labels/data"][start:stop], axis=-1)
        t_res = self.hf[self.regime + "/t_res/data"][start:stop]
        true_labels = np.concatenate((true_labels, np.expand_dims(t_res, axis=-1)), axis=-1)
        if self.set_up_Q_lim:
            data[:, :, 0] = np.where(
                data[:, :, 0] > self.Q_up_lim_norm, self.Q_up_lim_norm, data[:, :, 0]
            )
        if self.set_low_Q_lim:
            (data, true_labels, mask_channel) = eliminate_low_Q(self, data, true_labels)
        data = np.concatenate((data, mask_channel), axis=-1)
        (labels, ws) = make_2cl_labels(
            true_labels, self.relabel_big_tres, self.time_limit, self.weights
        )
        # apply noise
        if self.apply_add_gauss:
            (data, labels, ws) = add_gauss(data, labels, ws, self.g_add_stds)
        if self.apply_mult_gauss:
            data[:, :, 0] = self.mult_gauss.make_noise(data[:, :, 0], data[:, :, -1])
        return (data, labels, ws)

    def __call__(self):
        start = self.buffer_size
        stop = self.buffer_size + self.batch_size
        (buffer_data, buffer_labels, buffer_ws) = self.step(0, self.buffer_size)
        for i in range(self.batch_num):
            idxs = rd.sample(range(self.buffer_size), k=self.batch_size)
            yield (buffer_data[idxs], buffer_labels[idxs], buffer_ws[idxs])
            (data, labels, ws) = self.step(start, stop)
            buffer_data[idxs] = data
            buffer_labels[idxs] = labels
            buffer_ws[idxs] = ws
            start += self.batch_size
            stop += self.batch_size
        # fill the buffer with left data, if any
        (data, labels, ws) = self.step(start, stop)
        buffer_data = np.concatenate((buffer_data, data), axis=0)
        buffer_labels = np.concatenate((buffer_labels, labels), axis=0)
        buffer_ws = np.concatenate((buffer_ws, ws), axis=0)
        sh_idxs = rd.sample(range(buffer_labels.shape[0]), k=buffer_labels.shape[0])
        start = 0
        stop = self.batch_size
        for i in range(self.last_batches_num):
            idxs = sh_idxs[start:stop]
            yield (buffer_data[idxs], buffer_labels[idxs], buffer_ws[idxs])
            start += self.batch_size
            stop += self.batch_size


### due to technical reason (tf side), bs is constant and no reminder
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
    if make_generator_shuffle:
        tr_generator = generator_with_shuffle(
            h5f,
            "train",
            train_batch_size,
            train_buffer_size,
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
    else:
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
            tf.TensorSpec(shape=(tr_batch_size, max_len, 6)),
            tf.TensorSpec(shape=(tr_batch_size, max_len, 3)),
            tf.TensorSpec(shape=(tr_batch_size, max_len)),
        ),
    )

    if make_generator_shuffle:
        train_dataset = train_dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.repeat(-1).shuffle(10)

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
            tf.TensorSpec(shape=(tr_batch_size, max_len, 6)),
            tf.TensorSpec(shape=(tr_batch_size, max_len, 3)),
            tf.TensorSpec(shape=(tr_batch_size, max_len)),
        ),
    )

    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
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
            tf.TensorSpec(shape=(val_batch_size, max_len, 6)),
            tf.TensorSpec(shape=(val_batch_size, max_len, 3)),
            tf.TensorSpec(shape=(val_batch_size, max_len)),
        ),
    )

    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    return val_dataset
