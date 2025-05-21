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
from Configuration import CPU_NUMBER

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
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training configurations
    trainer: Trainer
    learning_rate: float
    max_epochs: int
    optimizer: Any
    loss_function: Any

    # Hardware configurations
    seed: Optional[int]
    devices: Optional[int]

    def __init__(
            self,
            model_name: str,
            learning_rate: float = 0.001,
            max_epochs: int = 10,
            optimizer: str = "adam",
            loss_function: str = "cross_entropy",
            seed: Optional[int] = 42,
            devices: Optional[int] = None
    ):
        super().__init__()
        self.model_name = model_name
        
        # Training configurations
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
        self.devices = devices if devices else CPU_NUMBER
        
        self._configure_logging()
        self._configure_trainer()

    def _configure_logging(self):
        self.log_service = LoggingService()
        self.root_dir = self.log_service.get_root_dir()
        os.makedirs(f"{self.root_dir}/{self.model_name}", exist_ok=False)

    def _configure_trainer(self):
        logger = TensorBoardLogger(f"{self.root_dir}/tensorboard", name=self.model_name)

        # TODO What is save top_k?? Should it be 3??
        # TODO What is checkpoint_dir and checkpoint_callback=True??
        self.trainer = Trainer(
            accelerator="cpu", 
            devices=self.devices,
            strategy="auto",
            max_epochs=self.max_epochs,
            logger=logger,
            default_root_dir=self.root_dir
        )

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented")
    
    def on_train_epoch_end(self):
        self.history["train_loss"].append(self.trainer.logged_metrics['train_loss'])
        self.history["train_acc"].append(self.trainer.logged_metrics['train_acc'])
        self.history['val_loss'].append(self.trainer.logged_metrics['val_loss'])
        self.history['val_acc'].append(self.trainer.logged_metrics['val_acc'])

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
        
        self.log_dict({"learning_rate":self.learning_rate}, on_step=False, on_epoch=True, prog_bar=True)
        
        # if batch_idx % 20 == 0:
        #     selected_x = x[:8]
        #     grid = torchvision.utils.make_grid(selected_x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("input_images", grid, 0)

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
        self.log_service.save_model(self)

    # TODO Not sure is working at the moment
    def load_model(self, file_path):
        state_dict = load(file_path)
        self.load_state_dict(state_dict)