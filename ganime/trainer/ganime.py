import os
from omegaconf import OmegaConf
import omegaconf
from ray.tune import Trainable
from ganime.model.base import load_model
from ganime.data.base import load_dataset
import tensorflow as tf

from ganime.utils.callbacks import TensorboardImage


class TrainableGANime(Trainable):
    def setup(self, config):
        strategy = tf.distribute.MirroredStrategy()

        tune_config = self.load_config_file_and_replace(config)
        self.batch_size = tune_config["trainer"]["batch_size"]

        self.n_devices = strategy.num_replicas_in_sync
        self.global_batch_size = self.batch_size * self.n_devices

        self.train_dataset, self.validation_dataset, self.test_dataset = load_dataset(
            dataset_name=config["dataset_name"],
            dataset_path=config["dataset_path"],
            batch_size=self.global_batch_size,
        )

        self.model = load_model(config["model"], config=tune_config, strategy=strategy)

        for data in self.train_dataset.take(1):
            train_sample_data = data
        for data in self.validation_dataset.take(1):
            validation_sample_data = data

        tensorboard_image_callback = TensorboardImage(
            self.logdir, train_sample_data, validation_sample_data
        )
        checkpointing = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.logdir, "checkpoint", "checkpoint"),
            monitor="total_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        self.callbacks = [tensorboard_image_callback, checkpointing]

    def load_config_file_and_replace(self, config):
        cfg = OmegaConf.load(config["config_file"])
        hyperparameters = config["hyperparameters"]

        for hp_key, hp_value in hyperparameters.items():
            cfg = self.replace_item(cfg, hp_key, hp_value)
        return cfg

    def replace_item(self, obj, key, replace_value):
        for k, v in obj.items():
            if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
                obj[k] = self.replace_item(v, key, replace_value)
        if key in obj:
            obj[key] = replace_value
        return obj

    def step(self):

        self.model.fit(
            self.train_dataset,
            initial_epoch=self.training_iteration,
            epochs=self.training_iteration + 1,
            callbacks=self.callbacks,
            verbose=0,
        )
        scores = self.model.evaluate(self.validation_dataset, verbose=0)
        return dict(zip(self.model.metrics_names, scores))

    def save_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        # return tmp_checkpoint_dir
        pass

    def load_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # self.model.load_state_dict(torch.load(checkpoint_path))
        pass
