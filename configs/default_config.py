from ml_collections.config_dict import ConfigDict


def get_config():
    cfg = ConfigDict()

    training = cfg.training = ConfigDict()
    training.batch_size = 2048
    training.max_steps = 20000
    training.log_freq = 100
    training.eval_freq = 1000
    training.ckpt_freq = 1000

    model = cfg.model = ConfigDict()
    model.name = "gaussian3d"
    model.target_compression_ratio = 64.0
    model.num_gaussians = 0
    model.truncation_sigma = 3.0
    model.init_scale = 0.05
    model.coordinate_bits = 16
    model.scale_bits = 8
    model.intensity_bits = 8
    model.quantization_enabled = True
    model.min_scale = 1e-3
    model.max_scale = 1.0
    model.intensity_range = 1.0
    model.clip_grad_norm = 1.0
    model.quantizer_overhead_bits = 256
    model.forward_query_chunk_size = 2048
    model.forward_gaussian_chunk_size = 4096

    optim = cfg.optim = ConfigDict()
    optim.optimizer = "adamw"
    optim.schedule = "cosineannealinglr"
    optim.initial_lr = 1e-3
    optim.weight_decay = 1e-4
    optim.min_lr = 1e-5
    optim.warmup_steps = 0

    data = cfg.data = ConfigDict()
    data.task = "pam"
    data.path = ""
    data.normalize = "minmax"
    data.coord_norm = 1.0

    eval_cfg = cfg.eval = ConfigDict()
    eval_cfg.chunk_size = 262144

    cfg.seed = 42
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg
