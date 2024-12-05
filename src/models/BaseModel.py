from pytorch_lightning import LightningModule
from utils import calculate_accuracy
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger

from typing import Any, Dict, Optional
from pytorch_lightning import Trainer
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from torch import load, save
import os

from LoggingService import LoggingService

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

        # Hardware configurations
        self.seed = seed
        self.gpus = gpus
        
        self._configure_logging()
        self._configure_trainer()

    def _configure_logging(self):
        self.log_service = LoggingService()
        self.root_dir = self.log_service.get_root_dir()
        os.makedirs(f"{self.root_dir}/{self.model_name}", exist_ok=True)
        
    def _configure_trainer(self):
        logger = self.log_service.create_tb_logger(self.model_name)

        # TODO What is save top_k?? Should it be 3??
        # TODO What is checkpoint_dir and checkpoint_callback=True??
        # TODO How to use the GPUs
        self.trainer = Trainer(
            accelerator="auto", 
            devices="auto",
            strategy="auto",
            max_epochs=self.max_epochs,
            logger=logger,
            default_root_dir=self.root_dir
        )

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)

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
            prog_bar=False)
        
        if batch_idx % 20 == 0:
            selected_x = x[:8]
            grid = torchvision.utils.make_grid(selected_x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("input_images", grid, 0)

        return loss

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
            prog_bar=False)

        return loss

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
            prog_bar=False)
        
        return loss
    
    def fit(self, train_dataloaders, val_dataloaders):
        self.trainer.fit(self, train_dataloaders, val_dataloaders)

    # TODO Not sure is working at the moment
    def load_model(self, file_path):
        state_dict = load(file_path)
        self.load_state_dict(state_dict)

    # TODO Still under construction
    def save_model(self, model_name, model_instance):
        """Save the model to the specified path."""
        model_file = f"{self.model_dir}/model.pt"
        save(model_instance.state_dict(), model_file)
        print(f"Model '{model_name}' saved at '{model_file}'.")