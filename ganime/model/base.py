import tensorflow as tf
from ganime.model.vqgan_clean.vqgan import VQGAN


def load_model(
    model: str, config: dict, strategy: tf.distribute.Strategy
) -> tf.keras.Model:

    if model == "vqgan":
        with strategy.scope():
            print(config["model"])
            model = VQGAN(**config["model"])
            gen_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
            disc_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            model.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
        return model
    else:
        raise ValueError(f"Unknown model: {model}")

    # if model == "moving_vae":
    #     from ganime.model.moving_vae import MovingVAE

    #     with strategy.scope():
    #         model = MovingVAE(input_shape=input_shape)

    #         negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    #         model.compile(
    #             optimizer=tf.optimizers.Adam(learning_rate=config["lr"]),
    #             loss=negloglik,
    #         )
    #         # model.build(input_shape=(None, *input_shape))
    #         # model.summary()

    #     return model
