from ray.tune import Trainable
from ganime.data.base import load_dataset
from ganime.model.base import load_model


class TrainableGANime(Trainable):
    def setup(self, config):
        self.dataset = load_dataset(config["dataset"])
        self.model = load_model(config["model"])
        self.epochs = config["epochs"]

    def step(self):
        for _ in range(self.epochs):
            self.model.fit(self.dataset)
        # score = objective(self.x, self.a, self.b)
        # self.x += 1
        # return {"score": score}
        pass

    def save_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        # return tmp_checkpoint_dir
        pass

    def load_checkpoint(self, tmp_checkpoint_dir):
        # checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        # self.model.load_state_dict(torch.load(checkpoint_path))
        pass
