from configs.gaussian3d import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.data.path = "data/smoke_volume.npy"
    cfg.model.target_compression_ratio = 10.0
    cfg.training.max_steps = 2
    cfg.training.batch_size = 128
    cfg.training.log_freq = 1
    cfg.training.eval_freq = 2
    cfg.training.ckpt_freq = 2
    cfg.training.patch_loss_enable = True
    cfg.training.patch_batch_size = 1
    cfg.training.patch_size = (8, 8, 8)
    cfg.training.patch_lambda_grad = 0.05
    cfg.eval.chunk_size = 256
    return cfg
