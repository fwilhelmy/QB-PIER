import torch
from models.BaseModel import BaseModel

class Arguments:
    # Experiment
    exp_name: str = "default"
    seed: int = 42 
    verbose: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator: str = "auto"
    devices: int = 1
    num_workers: int = 0
    # Optional override for PyTorch's thread count. Setting this can
    # improve reproducibility and limit CPU usage during training.
    num_threads: int | None = None

    # Model
    model: BaseModel
    model_config: dict

    # Data
    dataset: str = "MNIST"  # [MNIST, FMNIST] only for now..

    train_ratio: float = 0.6
    eval_ratio: float = 0.4

    train_batch_size: int = 512
    eval_batch_size: int = 2**12

    train_log_interval: float = 0.5
    eval_log_interval: float = 0.5
    
    # Optimization
    n_steps : int = 10**4 + 1
    optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
    scheduler: str = 'dummy'  # [dummy, step, cosine, plateau]
    loss_fn: str = 'cross_entropy'  # [cross_entropy, mse]
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1.0


def arguments_from_config(cfg) -> "Arguments":
    """Create an :class:`Arguments` instance from a Hydra config."""
    args = Arguments()

    # Basic experiment
    args.exp_name = cfg.model.exp_name
    args.verbose = cfg.get("verbose", False)

    # Device
    args.device = cfg.device.device
    args.accelerator = cfg.device.accelerator
    args.devices = cfg.device.devices
    args.num_workers = cfg.device.num_workers
    args.num_threads = cfg.device.num_threads

    # Dataset
    args.dataset = cfg.dataset.dataset
    args.train_ratio = cfg.dataset.train_ratio
    args.eval_ratio = cfg.dataset.eval_ratio
    args.train_batch_size = cfg.dataset.train_batch_size
    args.eval_batch_size = cfg.dataset.eval_batch_size
    input_shape = cfg.dataset.input_shape
    num_classes = cfg.dataset.classes

    # Training settings
    args.optimizer = cfg.training.optimizer
    args.scheduler = cfg.training.scheduler
    args.loss_fn = cfg.training.loss_fn
    args.lr = cfg.training.lr
    args.momentum = cfg.training.momentum
    args.weight_decay = cfg.training.weight_decay
    args.n_steps = cfg.get("n_steps", args.n_steps)
    args.train_log_interval = cfg.get("train_log_interval", args.train_log_interval)
    args.eval_log_interval = cfg.get("eval_log_interval", args.eval_log_interval)

    # Model class and configuration
    args.model = cfg.model.class_obj
    args.model_config = dict(cfg.model.params)

    channels = int(input_shape[0])
    img_size = int(input_shape[-1])
    flattened = int(channels * img_size * img_size)

    if "input_size" in args.model_config:
        val = args.model_config["input_size"]
        if isinstance(val, (list, tuple)):
            args.model_config["input_size"] = (channels, img_size)
        else:
            args.model_config["input_size"] = flattened
    if "in_channels" in args.model_config:
        args.model_config["in_channels"] = channels
    if "num_classes" in args.model_config:
        args.model_config["num_classes"] = num_classes
    if "classes" in args.model_config:
        args.model_config["classes"] = num_classes

    return args
