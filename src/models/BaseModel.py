import pytorch_lightning as L
from utils import calculate_accuracy

# TODO Would early stopping be interesting??
# If so, how to implement it??

# TODO Implement predict_step as a basic, might need to implement more advanced predict_step in the quantum models since we need multiple shots, etc.

# Base class
class BaseModel(L.LightningModule):
    def __init__(self, configuration):
        super().__init__()
        self.learning_rate = configuration.init_lr
        self.loss_function = configuration.loss_function
        self.optimizer = configuration.optimiser

    def forward(self, x):
        raise NotImplementedError

    # The training loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Use self(x) to call the forward method

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        # TODO
        #self.log('step', self.trainer.current_epoch)
        #self.log('loss', {'train': loss})

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    # The validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    # The test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)