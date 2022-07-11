import os

import click
import omegaconf
import ray
from pyprojroot.pyprojroot import here
from ray import tune
from ray.train import Trainer
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch

from ganime.trainer.ganime import TrainableGANime


def get_metric_direction(metric: str):
    if "loss" in metric:
        return "min"
    else:
        raise ValueError(f"Unknown metric: {metric}")


def trial_name_id(trial):
    return f"{trial.trainable_name}"


def trial_dirname_creator(trial):
    return f"{trial.trial_id}"


def get_search_space(model):
    if model == "vqgan":
        return {
            "beta": tune.uniform(0.1, 1.0),
            "num_embeddings": tune.choice([128, 256]),
            "embedding_dim": tune.choice([128, 256, 512, 1024]),
            "z_channels": tune.choice([64, 128, 256]),
            "channels": tune.choice([64, 128, 256]),
            "attention_resolution": tune.choice([[16], [32], [16, 32]]),
            "batch_size": tune.choice([8, 16]),
            "dropout": tune.choice([0.0, 0.1, 0.2]),
        }


def tune_ganime(
    experiment_name: str,
    dataset_name: str,
    config_file: str,
    model: str,
    metric: str,
    epochs: int,
    num_samples: int,
    num_cpus: int,
    num_gpus: int,
    max_concurrent_trials: int,
):

    dataset_path = here("data")
    analysis = tune.run(
        TrainableGANime,
        name=experiment_name,
        search_alg=ConcurrencyLimiter(
            OptunaSearch(), max_concurrent=max_concurrent_trials
        ),
        scheduler=AsyncHyperBandScheduler(max_t=epochs, grace_period=5),
        metric=metric,
        mode=get_metric_direction(metric),
        num_samples=num_samples,
        stop={"training_iteration": epochs},
        local_dir="./ganime_results",
        config={
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "model": model,
            "config_file": config_file,
            "hyperparameters": get_search_space(model),
        },
        resources_per_trial={
            "cpu": num_cpus // max_concurrent_trials,
            "gpu": num_gpus / max_concurrent_trials,
        },
        trial_name_creator=trial_name_id,
        trial_dirname_creator=trial_dirname_creator,
    )
    best_loss = analysis.get_best_config(metric="total_loss", mode="min")
    # best_accuracy = analysis.get_best_config(metric="accuracy", mode="max")
    print(f"Best loss config: {best_loss}")
    # print(f"Best accuracy config: {best_accuracy}")
    return analysis


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["moving_mnist_images", "kny_images"], case_sensitive=False),
    default="kny_images",
    help="Dataset to use",
)
@click.option(
    "--model",
    type=click.Choice(["vqgan", "net2net"], case_sensitive=False),
    default="vqgan",
    help="Model to use",
)
@click.option(
    "--epochs",
    default=300,
    help="Number of epochs to run",
)
@click.option(
    "--num_samples",
    default=50,
    help="Total number of trials to run",
)
@click.option(
    "--num_cpus",
    default=64,
    help="Number of cpus to use",
)
@click.option(
    "--num_gpus",
    default=8,
    help="Number of gpus to use",
)
@click.option(
    "--max_concurrent_trials",
    default=8,
    help="Maximum number of concurrent trials",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["total_loss", "reconstruction_loss", "vq_loss", "disc_loss"],
        case_sensitive=False,
    ),
    default="total_loss",
    help="The metric used to select the best trial",
)
@click.option(
    "--experiment_name",
    default="kny_images",
    help="The name of the experiment for logging in Tensorboard",
)
@click.option(
    "--config_file",
    default="kny_image.yaml",
    help="The name of the config file located inside ./config",
)
def run(
    experiment_name: str,
    config_file: str,
    dataset: str,
    model: str,
    epochs: int,
    num_samples: int,
    num_cpus: int,
    num_gpus: int,
    max_concurrent_trials: int,
    metric: str,
):
    config_file = here(os.path.join("configs", config_file))

    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    tune_ganime(
        experiment_name=experiment_name,
        dataset_name=dataset,
        config_file=config_file,
        model=model,
        epochs=epochs,
        num_samples=num_samples,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        max_concurrent_trials=max_concurrent_trials,
        metric=metric,
    )
