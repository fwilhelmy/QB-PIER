"""Hybrid model composed of interchangeable encoders and classifiers."""

from __future__ import annotations

import torch
from torch import Tensor

from models import BaseModel, ModelFactory
from .encoders import ENCODER_REGISTRY
from .classifiers import CLASSIFIER_REGISTRY


@ModelFactory.model("hqnn")
class HQNN(BaseModel):
    """Hybrid Quantum Neural Network (HQNN)."""

    def __init__(
        self,
        encoder: str,
        classifier: str,
        encoder_params: dict | None = None,
        classifier_params: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        encoder_params = encoder_params or {}
        classifier_params = classifier_params or {}
        self.save_hyperparameters(
            "encoder", "classifier", "encoder_params", "classifier_params"
        )

        encoder_cls = ENCODER_REGISTRY[encoder]
        self.encoder = encoder_cls(self.input_size, **encoder_params)

        classifier_cls = CLASSIFIER_REGISTRY[classifier]
        self.classifier = classifier_cls(
            in_features=self.encoder.output_size,
            output_size=self.output_size,
            **classifier_params,
        )

    # ------------------------------------------------------------------
    # Convenience methods to manipulate encoder and classifier
    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_classifier(self) -> None:
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_classifier(self) -> None:
        for p in self.classifier.parameters():
            p.requires_grad = True

    def train_encoder(self) -> None:
        self.unfreeze_encoder()
        self.freeze_classifier()

    def train_classifier(self) -> None:
        self.unfreeze_classifier()
        self.freeze_encoder()

    def swap_encoder(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder

    def swap_classifier(self, classifier: torch.nn.Module) -> None:
        self.classifier = classifier

    def load_encoder(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(state)

    def load_classifier(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.classifier.load_state_dict(state)

    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D401
        """Process inputs through encoder and classifier."""
        x = self.encoder(inputs)
        return self.classifier(x)


__all__ = ["HQNN"]
