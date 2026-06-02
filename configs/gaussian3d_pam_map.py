from configs.gaussian3d import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.data.path = "/home/shangqing.tong/pam/image-compression-torch/data/pam/mouse_ear/mouse_ear_001_1024x512x180.npy"
    cfg.training.max_steps = 200000
    cfg.training.batch_size = 2048
    cfg.training.log_freq = 250
    cfg.training.eval_freq = -1
    cfg.training.ckpt_freq = -1
    cfg.training.map_loss_enable = True
    cfg.training.map_loss_start_step = 2000
    cfg.training.map_loss_type = "soft"
    cfg.training.map_loss_weight = 0.05
    cfg.training.map_grad_loss_weight = 0.01
    cfg.training.map_softmax_tau = 10.0
    cfg.training.map_topk = 4
    cfg.training.map_column_sample_height = 32
    cfg.training.map_column_sample_width = 32
    cfg.model.target_compression_ratio = 512
    cfg.eval.chunk_size = 262144
    return cfg
