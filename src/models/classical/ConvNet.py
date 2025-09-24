import torch
from torch.nn import (
    Conv2d, Linear, MaxPool2d, ReLU, Sequential
)

from models import BaseModel, ModelFactory

@ModelFactory.model("convnet")
class ConvNet(BaseModel):
    """Simple configurable convolutional network."""
    def __init__(self, conv_channels=(32, 64), fc_sizes=(128,), **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters("conv_channels", "fc_sizes")
        in_channels, img_size, _ = self.input_size
        layers = []
        for out_ch in conv_channels:
            layers.append(Conv2d(in_channels, out_ch, kernel_size=3, padding=1))
            layers.append(ReLU())
            layers.append(MaxPool2d(2))
            img_size //= 2
            in_channels = out_ch
        self.features = Sequential(*layers)

        flat_dim = in_channels * img_size * img_size
        fc_layers = []
        prev = flat_dim
        for hidden in fc_sizes:
            fc_layers.append(Linear(prev, hidden))
            fc_layers.append(ReLU())
            prev = hidden
        fc_layers.append(Linear(prev, self.output_size))
        self.classifier = Sequential(*fc_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
