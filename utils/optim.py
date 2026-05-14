import torch


def get_optim(model, config):
    optimizer_name = config.optim.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optim.initial_lr,
            weight_decay=config.optim.weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optim.initial_lr,
            weight_decay=config.optim.weight_decay,
        )
    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=config.optim.initial_lr,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"{config.optim.optimizer} is not supported.")

    schedule_name = config.optim.schedule.lower()
    if schedule_name == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_steps,
            eta_min=config.optim.min_lr,
        )
    elif schedule_name == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=config.training.max_steps,
        )
    else:
        raise ValueError(f"{config.optim.schedule} is not supported.")

    return optimizer, scheduler


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
