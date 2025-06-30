import torch
from models.BaseModel import BaseModel

class Arguments:
    # Experiment
    exp_name: str = "default"
    seed: int = 42 
    verbose: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

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
    weight_decay: float = 1e-0