"""Hybrid parallel CNN combined with a quantum layer."""

from __future__ import annotations

import torch
from torch import Tensor, flatten
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Linear,
    LogSoftmax,
    MaxPool2d,
    Module,
    ReLU,
)

from models import BaseModel, ModelFactory
from models.quantum import ParallelVQC


class Encoder(Module):
    """Convolutional feature extractor for the HQNN."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        c1, c2 = conv_channels

        self.conv1 = Conv2d(
            in_channels=input_size[0],
            out_channels=c1,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm1 = BatchNorm2d(c1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm2 = BatchNorm2d(c2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self._out_features = self._get_conv_out_size()

    def _get_conv_out_size(self) -> int:
        """Compute the flattened feature size after the convolutional stack."""
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size)
            x = self.conv1(dummy)
            x = self.batchNorm1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.batchNorm2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            return x.view(1, -1).size(1)

    @property
    def output_size(self) -> int:
        """Size of the flattened output features."""
        return self._out_features

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the convolutional stack and flatten the result."""
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        return flatten(x, 1)


class Classifier(Module):
    """Fully-connected and quantum layers producing class logits."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        qblock_size: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(in_features=in_features, out_features=in_qlayer)
        self.relu = ReLU()

        self.qlayer = ParallelVQC(
            input_size=in_qlayer,
            block_size=qblock_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
        )
        self.fc2 = Linear(
            in_features=self.qlayer.output_size, out_features=output_size
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the classical and quantum classifier layers."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return self.logSoftmax(x)


@ModelFactory.model("hqnn")
class HQNN(BaseModel):
    """Hybrid Quantum Neural Network (HQNN)."""

    def __init__(
        self,
        qblock_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        conv_channels: tuple[int, int] = (16, 32),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "qblock_size", "in_qlayer", "n_qdepth", "ansatz", "conv_channels"
        )

        self.encoder = Encoder(self.input_size, conv_channels)
        self.classifier = Classifier(
            in_features=self.encoder.output_size,
            output_size=self.output_size,
            in_qlayer=in_qlayer,
            qblock_size=qblock_size,
            n_qdepth=n_qdepth,
            ansatz=ansatz,
        )

    # ------------------------------------------------------------------
    # Convenience methods to manipulate encoder and classifier
    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_classifier(self) -> None:
        """Freeze the classifier parameters."""
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_classifier(self) -> None:
        for p in self.classifier.parameters():
            p.requires_grad = True

    def train_encoder(self) -> None:
        """Train only the encoder."""
        self.unfreeze_encoder()
        self.freeze_classifier()

    def train_classifier(self) -> None:
        """Train only the classifier."""
        self.unfreeze_classifier()
        self.freeze_encoder()

    def swap_encoder(self, encoder: Encoder) -> None:
        """Swap the current encoder for a new one."""
        self.encoder = encoder

    def swap_classifier(self, classifier: Classifier) -> None:
        """Swap the current classifier for a new one."""
        self.classifier = classifier

    def load_encoder(self, path: str) -> None:
        """Load encoder weights from a file."""
        state = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(state)

    def load_classifier(self, path: str) -> None:
        """Load classifier weights from a file."""
        state = torch.load(path, map_location="cpu")
        self.classifier.load_state_dict(state)

    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D401
        """Process inputs through encoder and classifier."""
        x = self.encoder(inputs)
        return self.classifier(x)


__all__ = ["HQNN", "Encoder", "Classifier"]

