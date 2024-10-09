from pytorch_lightning import LightningModule
from utils import calculate_accuracy
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger

from typing import Any, Dict, Optional
from pytorch_lightning import Trainer
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn, load

SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss, # For classification tasks
    "mse": MSELoss # For regression tasks
}

# Base class
class BaseModel(LightningModule):
    # General configurations
    model_name: str

    # Training configurations
    trainer: Trainer
    batch_size: int
    learning_rate: float
    max_epochs: int
    optimizer: Any
    loss_function: Any

    # Logging configurations
    log_dir: str
    checkpoint_dir: str

    # Hardware configurations
    seed: Optional[int]
    gpus: Optional[int]

    def __init__(
            self,
            model_name: str,
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
        super().__init__()
        self.model_name = model_name
        
        # Training configurations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        if optimizer.lower() not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        self.optimizer_class = SUPPORTED_OPTIMIZERS[optimizer.lower()]

        if loss_function.lower() not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        self.loss_function = SUPPORTED_LOSS_FUNCTIONS[loss_function.lower()]()

        # Logging configuration
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Hardware configurations
        self.seed = seed
        self.gpus = gpus

        self.trainer = self._init_trainer()

    def _init_trainer(self):
        logger = TensorBoardLogger(self.log_dir, name="logs_test_" + self.model_name)

        # TODO What is save top_k?? Should it be 3??
        # TODO What is checkpoint_dir and checkpoint_callback=True??
        # TODO How to use the GPUs
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            default_root_dir=self.log_dir
        )

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented")

    # The training loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Use self(x) to call the forward method

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict({
            "train_loss": loss, 
            "train_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True)
        
        if batch_idx % 20 == 0:
            selected_x = x[:8]
            grid = torchvision.utils.make_grid(selected_x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("input_images", grid, 0)

    # The validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict({
            "val_loss": loss, 
            "val_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True)

    # The test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)
        
        self.log_dict({
            "test_loss": loss, 
            "test_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)
    
    def get_trainer(self):
        # TODO self.logger_rootdir in LogsManager now, should call the get_logger
        logger = TensorBoardLogger(self.logger_rootdir, name="logs_test_" + self.model_name)

        # TODO What is save top_k?? Should it be 3??
        # TODO What is checkpoint_dir and checkpoint_callback=True??
        return Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            logger=logger,
            default_root_dir=self.logger_rootdir
        )
    
    # TODO Not sure is working at the moment
    def load_model(self, file_path):
        state_dict = load(file_path)
        self.load_state_dict(state_dict)