import torch
from models import BaseModel

class Arguments:
    # Experiment
    exp_name: str = "default"
    exp_id: int = 0

    # Model
    model: BaseModel

    # Data
    train_samples: int = 10**4
    eval_samples: int = 10**4

    train_batch_size: int = 512
    eval_batch_size: int = 2**12
    num_workers: int = 0
    
    # Optimization
    n_steps : int = 10**4 + 1
    optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
    scheduler: str = 'dummy'  # [dummy, step, cosine, plateau]
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-0

    # Logging
    eval_start: int = 100
    eval_step: int = 100
    save_step: int = 1000