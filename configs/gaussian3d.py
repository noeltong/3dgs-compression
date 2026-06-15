from configs.default_config import get_config as get_default_config


def get_config():
    cfg = get_default_config()
    cfg.training.max_steps = 10000
    cfg.training.log_freq = 50
    cfg.training.eval_freq = 500
    cfg.training.ckpt_freq = 500
    cfg.model.target_compression_ratio = 64.0
    cfg.model.coordinate_bits = 16
    cfg.model.scale_bits = 8
    cfg.model.intensity_bits = 8
    return cfg
