from typing import Any, Dict, Optional
from models.BaseModel import BaseModel
from pytorch_lightning import Trainer
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss

# Supported optimizers and loss functions
SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss
}

class ModelConfiguration:


    def __init__(
        self,
        model_name: str,
        trainer: Trainer,
        model: BaseModel,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_epochs: int = 10,
        optimizer: str = "adam",
        loss_function: str = "cross_entropy",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        save_top_k: int = 3,
        monitor_metric: str = "val_loss",
        seed: Optional[int] = 42,
        gpus: Optional[int] = None  # Set to None for CPU training
    ):
        # Training configuration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Instantiate optimizer
        if optimizer.lower() in SUPPORTED_OPTIMIZERS:
            self.optimizer = SUPPORTED_OPTIMIZERS[optimizer.lower()]
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Instantiate loss function
        if loss_function.lower() in SUPPORTED_LOSS_FUNCTIONS:
            self.loss_function = SUPPORTED_LOSS_FUNCTIONS[loss_function.lower()]()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Logging configuration
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric

        # Hardware configurations
        self.seed = seed
        self.gpus = gpus

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration into a dictionary format for easy access and logging."""
        return self.__dict__