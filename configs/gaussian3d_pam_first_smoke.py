from configs.gaussian3d_pam_first import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.model.target_compression_ratio = 100000.0
    cfg.training.max_steps = 2
    cfg.training.batch_size = 128
    cfg.training.log_freq = 1
    cfg.training.eval_freq = -1
    cfg.training.ckpt_freq = -1
    cfg.eval.chunk_size = 4096
    cfg.model.forward_query_chunk_size = 1024
    cfg.model.forward_gaussian_chunk_size = 512
    return cfg
