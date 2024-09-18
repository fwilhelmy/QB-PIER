from pytorch_lightning import LightningModule
from utils import calculate_accuracy
import torchvision
from models.ModelConfiguration import ModelConfiguration

# TODO Would early stopping be interesting??
# If so, how to implement it??

# TODO Implement predict_step as a basic, might need to implement more advanced predict_step in the quantum models since we need multiple shots, etc.

# Base class
class BaseModel(LightningModule):
    def __init__(self, model_name: str, configuration: ModelConfiguration):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = configuration.learning_rate
        self.loss_function = configuration.loss_function
        self.optimizer_class = configuration.optimizer

    def forward(self, x):
        raise NotImplementedError

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