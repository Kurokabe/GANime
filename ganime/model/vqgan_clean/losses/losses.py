import tensorflow as tf
from tensorflow.keras.losses import (
    MeanSquaredError,
    Reduction,
    SparseCategoricalCrossentropy,
)
from tensorflow import reduce_mean
from tensorflow.keras.losses import Loss
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import Model


class Losses:
    def __init__(self, num_replicas):
        self.num_replicas = num_replicas
        self.SCCE = SparseCategoricalCrossentropy(
            from_logits=True, reduction=Reduction.NONE
        )
        self.MSE = MeanSquaredError(reduction=Reduction.NONE)

        self.vgg = VGG.build()
        self.preprocess = preprocess_input

    def perceptual_loss(self, real, pred):
        y_true_preprocessed = self.preprocess(real)
        y_pred_preprocessed = self.preprocess(pred)
        y_true_scaled = y_true_preprocessed / 12.75
        y_pred_scaled = y_pred_preprocessed / 12.75

        loss = self.mse_loss(y_true_scaled, y_pred_scaled) * 5e3

        return loss

    def scce_loss(self, real, pred):
        # compute categorical cross entropy loss without reduction
        loss = self.SCCE(real, pred)
        # compute reduced mean over the entire batch
        loss = reduce_mean(loss) * (1.0 / self.num_replicas)
        # return reduced scce loss
        return loss

    def mse_loss(self, real, pred):
        # compute mean squared error without reduction
        loss = self.MSE(real, pred)
        # compute reduced mean over the entire batch
        loss = reduce_mean(loss) * (1.0 / self.num_replicas)
        # return reduced mse loss
        return loss


class VGG:
    @staticmethod
    def build():
        # initialize the pre-trained VGG19 model
        vgg = VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        # slicing the VGG19 model till layer #20
        model = Model(vgg.input, vgg.layers[20].output)
        # return the sliced VGG19 model
        return model
