import tensorflow as tf


def load_model(model: str, input_shape: tuple, config: dict) -> tf.keras.Model:
    strategy = tf.distribute.MirroredStrategy()
    if model == "moving_vae":
        from ganime.model.moving_vae import MovingVAE

        with strategy.scope():
            model = MovingVAE(input_shape=input_shape)

            negloglik = lambda x, rv_x: -rv_x.log_prob(x)
            model.compile(
                optimizer=tf.optimizers.Adam(learning_rate=config["lr"]),
                loss=negloglik,
            )
            # model.build(input_shape=(None, *input_shape))
            # model.summary()

        return model
