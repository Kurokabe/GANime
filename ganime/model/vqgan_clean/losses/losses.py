import tensorflow as tf
from tensorflow.keras.losses import (
    MeanSquaredError,
    Reduction,
    SparseCategoricalCrossentropy,
)
from tensorflow import reduce_mean


class Losses:
    def __init__(self, num_replicas):
        self.num_replicas = num_replicas
        self.SCCE = SparseCategoricalCrossentropy(
            from_logits=True, reduction=Reduction.NONE
        )
        self.MSE = MeanSquaredError(reduction=Reduction.NONE)

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
