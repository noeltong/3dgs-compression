from configs.default_config import get_config as get_default_config


def get_config():
    cfg = get_default_config()
    cfg.training.max_steps = 10000
    cfg.training.log_freq = 50
    cfg.training.eval_freq = 500
    cfg.training.ckpt_freq = 500
    cfg.training.map_loss_enable = False
    cfg.training.map_loss_start_step = 0
    cfg.training.map_loss_type = "hard"
    cfg.training.map_loss_weight = 0.0
    cfg.training.map_grad_loss_weight = 0.0
    cfg.training.map_softmax_tau = 1.0
    cfg.training.map_topk = 8
    cfg.training.map_column_sample_mode = "patch"
    cfg.training.map_column_sample_height = 16
    cfg.training.map_column_sample_width = 16
    cfg.model.target_compression_ratio = 64.0
    cfg.model.coordinate_bits = 16
    cfg.model.scale_bits = 8
    cfg.model.intensity_bits = 8
    return cfg
