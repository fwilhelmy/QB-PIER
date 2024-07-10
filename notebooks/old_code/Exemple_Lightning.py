import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from DataManager import DataManager

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# dataset
data_manager = DataManager(dataset_name='MNIST')
train_loader, val_loader = data_manager.split_dataset(train_samples=500, val_samples=100, batch_size=64)

# model
# autoencoder = LitAutoEncoder(Encoder(), Decoder())
autoencoder = LitAutoEncoder.load_from_checkpoint("./logs/lightning_logs/version_1/checkpoints/epoch=9-step=80.ckpt")

# train model
trainer = L.Trainer(max_epochs=10, default_root_dir="logs/")
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)

# find where
# self.init_lr = init_lr
# self.batch_size = batch_size
# self.epochs = epochs
# self.device = get_device() if device is None else device
# self.model = model.to(self.device)
# self.optimizer = get_optimizer(self.model, init_lr)
# self.loss_fn = get_lossfn()