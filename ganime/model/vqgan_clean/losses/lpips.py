import os
import numpy as np
import tensorflow as tf
import torchvision.models as models
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.losses import Loss
from pyprojroot.pyprojroot import here


def normalize_tensor(x, eps=1e-10):
    norm_factor = tf.sqrt(tf.reduce_sum(x**2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)


class LPIPS(Loss):
    def __init__(self, use_dropout=True, **kwargs):
        super().__init__(**kwargs)

        self.scaling_layer = ScalingLayer()  # preprocess_input
        selected_layers = [
            "block1_conv2",
            "block2_conv2",
            "block3_conv3",
            "block4_conv3",
            "block5_conv3",
        ]

        # TODO here we load the same weights as pytorch, try with tensorflow weights
        self.net = self.load_vgg16()  # VGG16(weights="imagenet", include_top=False)
        self.net.trainable = False
        outputs = [self.net.get_layer(layer).output for layer in selected_layers]

        self.model = Model(self.net.input, outputs)
        self.lins = [NetLinLayer(use_dropout=use_dropout) for _ in selected_layers]

        # TODO: here we use the pytorch weights of the linear layers, try without these layers, or without initializing the weights
        self(tf.zeros((1, 16, 16, 1)), tf.zeros((1, 16, 16, 1)))
        self.init_lin_layers()

    def load_vgg16(self) -> Model:
        """Load a VGG16 model with the same weights as PyTorch
        https://github.com/ezavarygin/vgg16_pytorch2keras
        """
        pytorch_model = models.vgg16(pretrained=True)
        # select weights in the conv2d layers and transpose them to keras dim ordering:
        wblist_torch = list(pytorch_model.parameters())[:26]
        wblist_keras = []
        for i in range(len(wblist_torch)):
            if wblist_torch[i].dim() == 4:
                w = np.transpose(wblist_torch[i].detach().numpy(), axes=[2, 3, 1, 0])
                wblist_keras.append(w)
            elif wblist_torch[i].dim() == 1:
                b = wblist_torch[i].detach().numpy()
                wblist_keras.append(b)
            else:
                raise Exception("Fully connected layers are not implemented.")

        keras_model = VGG16(include_top=False, weights=None)
        keras_model.set_weights(wblist_keras)
        return keras_model

    def init_lin_layers(self):
        for i in range(5):
            weights = np.load(
                os.path.join(here(), "models", "NetLinLayer", f"numpy_{i}.npy")
            )
            weights = np.moveaxis(weights, 1, 2)
            self.lins[i].model.layers[1].set_weights([weights])

    def call(self, y_true, y_pred):

        scaled_true = self.scaling_layer(y_true)
        scaled_pred = self.scaling_layer(y_pred)

        outputs_true, outputs_pred = self.model(scaled_true), self.model(scaled_pred)
        features_true, features_pred, diffs = {}, {}, {}

        for kk in range(len(outputs_true)):
            features_true[kk], features_pred[kk] = normalize_tensor(
                outputs_true[kk]
            ), normalize_tensor(outputs_pred[kk])

            diffs[kk] = (features_true[kk] - features_pred[kk]) ** 2

        res = [
            tf.reduce_mean(self.lins[kk](diffs[kk]), axis=(-3, -2), keepdims=True)
            for kk in range(len(outputs_true))
        ]

        return tf.reduce_sum(res)


class ScalingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shift = tf.Variable([-0.030, -0.088, -0.188])
        self.scale = tf.Variable([0.458, 0.448, 0.450])

    def call(self, inputs):
        return (inputs - self.shift) / self.scale


class NetLinLayer(layers.Layer):
    def __init__(self, channels_out=1, use_dropout=False):
        super().__init__()
        sequence = (
            [
                layers.Dropout(0.5),
            ]
            if use_dropout
            else []
        )
        sequence += [
            layers.Conv2D(channels_out, 1, padding="same", use_bias=False),
        ]
        self.model = Sequential(sequence)

    def call(self, inputs):
        return self.model(inputs)
