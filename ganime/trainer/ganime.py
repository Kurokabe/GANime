from ray.tune import Trainable
from ganime.model.base import load_model
from ganime.data.base import load_dataset


class TrainableGANime(Trainable):
    def setup(self, config):
        self.train_dataset, self.test_dataset, input_shape = load_dataset(
            dataset_name=config["dataset_name"],
            dataset_path=config["dataset_path"],
            batch_size=config["batch_size"],
        )
        self.model = load_model(config["model"], input_shape=input_shape, config=config)

    def step(self):
        self.model.fit(self.train_dataset, epochs=1)
        score = self.model.evaluate(self.test_dataset)
        return {"loss": score}

    def save_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        # return tmp_checkpoint_dir
        pass

    def load_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # self.model.load_state_dict(torch.load(checkpoint_path))
        pass
