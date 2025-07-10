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
    dataset_root: str = "./data/processed"

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
    learning_rate: float = 1e-3
    optimiser_config: dict = {}


def arguments_from_config(cfg: dict) -> "Arguments":
    """Create an :class:`Arguments` instance from a configuration dict."""
    args = Arguments()

    # Basic experiment
    args.exp_name = cfg["model"]["model_name"]
    args.verbose = cfg.get("verbose", False)

    # Device
    args.device = cfg["device"]["device"]
    args.accelerator = cfg["device"]["accelerator"]
    args.devices = cfg["device"]["devices"]
    args.num_workers = cfg["device"]["num_workers"]
    args.num_threads = cfg["device"]["num_threads"]

    # Dataset
    args.dataset = cfg["dataset"]["dataset"]
    args.dataset_root = cfg["dataset"].get("root", cfg.get("DATA_DIR", args.dataset_root))
    args.train_ratio = cfg["dataset"]["train_ratio"]
    args.eval_ratio = cfg["dataset"]["eval_ratio"]
    args.train_batch_size = cfg["dataset"]["train_batch_size"]
    args.eval_batch_size = cfg["dataset"]["eval_batch_size"]
    
    # Training settings
    args.optimizer = cfg["training"]["optimizer"]
    args.scheduler = cfg["training"]["scheduler"]
    args.loss_fn = cfg["training"]["loss_fn"]
    args.learning_rate = cfg["training"]["learning_rate"]
    args.optimiser_config = cfg["training"].get("optimiser_config", {})
    args.n_steps = cfg.get("n_steps", args.n_steps)
    args.train_log_interval = cfg.get("train_log_interval", args.train_log_interval)
    args.eval_log_interval = cfg.get("eval_log_interval", args.eval_log_interval)

    # Model class and configuration
    args.model = cfg["model"]["class_obj"]
    args.model_config = dict(cfg["model"]["params"])
    args.model_config["input_size"] = cfg["dataset"]["input_shape"]
    args.model_config["output_size"] = cfg["dataset"]["output_size"]

    return args
