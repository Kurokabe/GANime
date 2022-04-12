import click
import ray
from ray.train import Trainer
from ray import tune
from ganime.trainer.ganime import TrainableGANime


def tune_ganime(dataset: str, model: str, num_workers: int, num_samples: int):
    analysis = tune.run(
        TrainableGANime,
        num_samples=num_samples,
        num_workers=num_workers,
        config={
            "dataset": dataset,
            "model": model,
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
            "epochs": 3,
        },
    )
    best_loss = analysis.get_best_config(metric="loss", mode="min")
    best_accuracy = analysis.get_best_config(metric="accuracy", mode="max")
    print(f"Best loss config: {best_loss}")
    print(f"Best accuracy config: {best_accuracy}")
    return analysis


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["MNIST"], case_sensitive=False),
    default="MNIST",
    help="Dataset to use",
)
@click.option(
    "--model",
    type=click.Choice(["P2P", "GANime"], case_sensitive=False),
    default="GANime",
    help="Model to use",
)
def run(dataset: str, model: str):
    ray.init()
    tune_ganime(dataset=dataset, model=model, num_workers=1, num_samples=1)
