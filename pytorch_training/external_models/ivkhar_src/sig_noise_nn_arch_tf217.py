import tensorflow as tf


### U-net like model
# encoder, downsamples 2x n times
def make_encoder_cnn(init_size, init_channels, n, filters, kernels, regularizations):
    data_in = tf.keras.Input(shape=(init_size, init_channels + 1))

    data = data_in[:, :, :-1]
    mask = data_in[:, :, -1:]

    encs = [data_in]

    assert len(filters) == n
    assert len(filters) == len(kernels)
    assert len(filters) == len(regularizations)

    x = data
    for fil, ker, reg in zip(filters, kernels, regularizations):
        x = tf.keras.layers.Conv1D(fil, ker, padding="same")(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = x * mask
        skip = x
        x = tf.keras.layers.Conv1D(fil, ker, padding="same")(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = x * mask
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip])
        x = tf.keras.layers.Conv1D(fil, ker, strides=2, padding="same")(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        mask = tf.keras.layers.MaxPool1D(pool_size=2, padding="same")(mask)
        x = x * mask
        dat = tf.keras.layers.Concatenate(axis=-1)([x, mask])
        encs.append(dat)

    model = tf.keras.Model(inputs=data_in, outputs=encs)
    return model


class getShape(tf.keras.layers.Layer):
    def call(self, x):
        return tf.shape(x)[1]


class reduceProper(tf.keras.layers.Layer):
    def call(self, x, skip):
        sh = tf.shape(skip)[1]
        return x[:, :sh, :]


# decoder, upsamples 2x n times
def make_decoder_cnn(n, filters, kernels, regularizations, shapes):
    getShaper = getShape()
    reducer = reduceProper()
    skips = [tf.keras.Input(shape=shape) for shape in shapes]

    assert len(filters) == n
    assert len(filters) == len(kernels)
    assert len(filters) == len(regularizations)
    assert len(filters) == len(shapes) - 1

    x = skips[0][:, :, :-1]
    mask = skips[0][:, :, -1:]

    for fil, ker, skip, reg in zip(filters, kernels, skips[1:], regularizations):
        x = tf.keras.layers.Conv1D(fil, ker, padding="same")(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = x * mask
        l_skip = x
        x = tf.keras.layers.Conv1D(fil, ker, padding="same")(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = x * mask
        x = tf.keras.layers.Concatenate(axis=-1)([x, l_skip])
        x = tf.keras.layers.Conv1DTranspose(
            fil, ker, strides=2, padding="same", output_padding=None
        )(x)
        x = tf.keras.layers.PReLU(shared_axes=[1], alpha_regularizer=tf.keras.regularizers.L2(reg))(
            x
        )
        x = tf.keras.layers.BatchNormalization()(x)
        # z = x[:,:skip.shape[1],:]
        # z = x[:,:getShaper(skip),:]
        z = reducer(x, skip)
        mask = skip[:, :, -1:]
        z = z * mask
        x = tf.keras.layers.Concatenate(axis=-1)([z, skip[:, :, :-1]])

    decs = x

    model = tf.keras.Model(inputs=skips, outputs=x)
    return model


class toBool(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x[:, :, -1], bool)


class predsWithMask(tf.keras.layers.Layer):
    def call(self, x, mask):
        return tf.where(
            tf.cast(mask, bool), tf.keras.layers.Softmax(axis=-1)(x), tf.constant([0.0, 1.0])
        )


# u-net model
def make_unet_model():
    data = tf.keras.Input(shape=(None, 6))

    to_bool_layer = toBool()
    mask_predictor = predsWithMask()

    mask_lstm = to_bool_layer(data)
    mask = data[:, :, -1:]

    # pre-analyze wtih rnn
    lstm_layer_pre = tf.keras.layers.LSTM(
        64, activation="tanh", recurrent_activation="sigmoid", return_sequences=True
    )
    bidir_pre = tf.keras.layers.Bidirectional(lstm_layer_pre, merge_mode="mul")

    x = bidir_pre(data, mask=mask_lstm)
    x = x * mask

    enc_filters = [80, 96, 48]
    enc_kernels = [12, 10, 8]
    enc_regs = [0.0, 0.0, 0.0]

    encoder = make_encoder_cnn(None, 64, 3, enc_filters, enc_kernels, enc_regs)
    x = tf.keras.layers.Concatenate(axis=-1)([x, mask])
    encs = encoder(x)

    dec_filters = [96, 112, 96]
    dec_kernels = [10, 12, 14]
    dec_regs = [0.0, 0.0, 0.0]

    rev = list(reversed(encs))
    shapes = [r.shape[1:] for r in rev]

    decoder = make_decoder_cnn(3, dec_filters, dec_kernels, dec_regs, shapes)
    x = decoder(rev)

    # post-rnn
    lstm_layer = tf.keras.layers.LSTM(
        64, activation="tanh", recurrent_activation="sigmoid", return_sequences=True
    )
    bidir = tf.keras.layers.Bidirectional(lstm_layer, merge_mode="mul")

    x = bidir(x, mask=mask_lstm)
    x = x * mask

    x = tf.keras.layers.Conv1D(2, 4, padding="same")(x)

    preds = mask_predictor(x, mask)

    model = tf.keras.Model(inputs=data, outputs=preds)
    return model
