from configs.gaussian3d import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.data.path = "/home/shangqing.tong/pam/image-compression-torch/data/pam/mouse_ear/mouse_ear_001_1024x512x180.npy"
    cfg.model.target_compression_ratio = 512
    cfg.training.max_steps = 200000
    cfg.training.log_freq = 250
    cfg.training.eval_freq = -1
    cfg.training.ckpt_freq = -1
    return cfg
