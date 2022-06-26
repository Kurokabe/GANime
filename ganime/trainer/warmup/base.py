from ganime.trainer.warmup.cosine import WarmUpCosine


def create_warmup_scheduler(trainer_config, num_devices):
    len_x_train = trainer_config["len_x_train"]
    batch_size = trainer_config["batch_size"]
    n_epochs = trainer_config["n_epochs"]

    total_steps = int(len_x_train / batch_size * n_epochs / num_devices)
    warmup_epoch_percentage = trainer_config["warmup_epoch_percentage"]
    warmup_steps = int(total_steps * warmup_epoch_percentage)

    scheduled_lrs = WarmUpCosine(
        lr_start=trainer_config["lr_start"],
        lr_max=trainer_config["lr_max"],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    return scheduled_lrs
