import click
import ray
from ray.train import Trainer
from ray import tune

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ganime.trainer.ganime import TrainableGANime
from pyprojroot.pyprojroot import here


def get_metric_direction(metric: str):
    if metric == "loss":
        return "min"
    elif metric == "accuracy":
        return "max"
    else:
        raise ValueError(f"Unknown metric: {metric}")


def trial_name_id(trial):
    return f"{trial.trainable_name}"


def trial_dirname_creator(trial):
    return f"{trial.trial_id}"


def tune_ganime(
    experiment_name: str,
    dataset_name: str,
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
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([128, 256]),
        },
        resources_per_trial={
            "cpu": num_cpus // max_concurrent_trials,
            "gpu": num_gpus / max_concurrent_trials,
        },
        trial_name_creator=trial_name_id,
        trial_dirname_creator=trial_dirname_creator,
    )
    best_loss = analysis.get_best_config(metric="loss", mode="min")
    # best_accuracy = analysis.get_best_config(metric="accuracy", mode="max")
    print(f"Best loss config: {best_loss}")
    # print(f"Best accuracy config: {best_accuracy}")
    return analysis


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["moving_mnist"], case_sensitive=False),
    default="moving_mnist",
    help="Dataset to use",
)
@click.option(
    "--model",
    type=click.Choice(["moving_vae", "P2P", "GANime"], case_sensitive=False),
    default="moving_vae",
    help="Model to use",
)
@click.option(
    "--epochs",
    default=5,
    help="Number of epochs to run",
)
@click.option(
    "--num_samples",
    default=4,
    help="Total number of trials to run",
)
@click.option(
    "--num_cpus",
    default=128,
    help="Number of cpus to use",
)
@click.option(
    "--num_gpus",
    default=8,
    help="Number of gpus to use",
)
@click.option(
    "--max_concurrent_trials",
    default=1,
    help="Maximum number of concurrent trials",
)
@click.option(
    "--metric",
    type=click.Choice(["loss"], case_sensitive=False),
    default="loss",
    help="The metric used to select the best trial",
)
@click.option(
    "--experiment_name",
    default="",
    help="The name of the experiment for logging in Tensorboard",
)
def run(
    experiment_name: str,
    dataset: str,
    model: str,
    epochs: int,
    num_samples: int,
    num_cpus: int,
    num_gpus: int,
    max_concurrent_trials: int,
    metric: str,
):
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    tune_ganime(
        experiment_name=experiment_name,
        dataset_name=dataset,
        model=model,
        epochs=epochs,
        num_samples=num_samples,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        max_concurrent_trials=max_concurrent_trials,
        metric=metric,
    )
