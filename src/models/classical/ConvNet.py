import torch
from torch import nn
from models.BaseModel import BaseModel

class ConvNet(BaseModel):
    """Simple configurable convolutional network."""
    def __init__(self, input_size=(3, 32), conv_channels=(32, 64), fc_sizes=(128,), num_classes=10, **kwargs):
        super().__init__(**kwargs)
        in_channels, img_size = input_size
        layers = []
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            img_size //= 2
            in_channels = out_ch
        self.features = nn.Sequential(*layers)

        flat_dim = in_channels * img_size * img_size
        fc_layers = []
        prev = flat_dim
        for hidden in fc_sizes:
            fc_layers.append(nn.Linear(prev, hidden))
            fc_layers.append(nn.ReLU())
            prev = hidden
        fc_layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
