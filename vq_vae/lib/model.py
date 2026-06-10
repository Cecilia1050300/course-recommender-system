import tensorflow as tf
from tensorflow.keras import layers

from my_lib import config as cfg
from my_lib import VectorQuantizer as VQ

def get_encoder(input_shape, latent_dim, num_layers, base_filters):
    inner_input = layers.Input(shape=input_shape)
    x = inner_input

    for i in range(num_layers):
        filters = base_filters * (2**i)
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)

    x = layers.Conv2D(latent_dim, 1, padding="same")(x)
    x = layers.LayerNormalization()(x)
    return tf.keras.Model(inner_input, x, name="encoder")

def get_decoder(token_shape_2d, latent_dim, num_layers, base_filters):
    inner_input = layers.Input(shape=(token_shape_2d[0], token_shape_2d[1], latent_dim))
    x = inner_input

    for i in range(num_layers):
        filters = base_filters * (2**(num_layers - i - 1))
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inner_input, x, name="decoder")

def build_vq_vae(num_embedding):
    encoder = get_encoder(cfg.input_shape, cfg.latent_dim, cfg.conv_layers, cfg.base_filter)
    decoder = get_decoder(cfg.token_shape_2D, cfg.latent_dim, cfg.conv_layers, cfg.base_filter)
    vq_layer = VQ.VectorQuantizer(num_embedding, cfg.latent_dim, commitment_cost=cfg.commitment_cost)

    inputs = layers.Input(shape=cfg.input_shape)
    enc_out = encoder(inputs)
    quantized = vq_layer(enc_out)
    reconstruction = decoder(quantized)

    vq_vae = tf.keras.Model(inputs, reconstruction, name="vq_vae")
    return vq_vae, encoder, decoder, vq_layer

def load_vq_vae(num_embedding):
    vq_vae, encoder, decoder, vq_layer = build_vq_vae(num_embedding)
    vq_vae.load_weights(cfg.n_dict[num_embedding][0])
    return vq_vae, encoder, decoder, vq_layer

def compile_and_train_vq_vae(vq_vae, train_dataset, num_embedding):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            cfg.n_dict[num_embedding][0],
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    vq_vae.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mse", "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    history = vq_vae.fit(train_dataset, epochs=cfg.epochs, callbacks=callbacks)
    return history, vq_vae