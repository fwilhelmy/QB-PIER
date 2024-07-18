from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from models.BaseModel import BaseModel
from pytorch_lightning import Trainer

@dataclass
class ModelWrapper:
    # Model configuration
    model_name: str
    trainer: Trainer
    model: BaseModel

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging and checkpointing
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    
    # Other configurations
    seed: Optional[int] = 42
    gpus: Optional[int] = None # Set to None for CPU training
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

# Example usage
config = Configuration()

# Print the configuration dictionary
print(config.to_dict())
