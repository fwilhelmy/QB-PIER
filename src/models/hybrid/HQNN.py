"""Hybrid model composed of interchangeable encoders and classifiers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Identity, Linear, LogSoftmax

from models import BaseModel, ModelFactory
from layers import LayerFactory


@ModelFactory.model("hqnn")
class HQNN(BaseModel):
    """Hybrid Quantum Neural Network (HQNN)."""

    def __init__(
        self,
        encoder: Mapping[str, Any] | DictConfig,
        classifier: Mapping[str, Any] | DictConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters("encoder", "classifier")

        encoder_name = encoder["name"]
        encoder_params = encoder.get("params", {}) or {}
        classifier_name = classifier["name"]
        classifier_params = classifier.get("params", {}) or {}

        classifier_output_size = classifier_params.pop("output_size", self.output_size)
        classifier_params.pop("apply_log_softmax", None)

        # If the classifier is configured to emit a different dimension than the
        # model's target ``output_size``, insert a lightweight projection before
        # the final log-softmax so that the logits are mapped correctly.
        if classifier_output_size != self.output_size:
            self.post = Linear(classifier_output_size, self.output_size, bias=True)
        else:
            self.post = Identity()
        self.output_activation = LogSoftmax(dim=1)

        self.encoder = LayerFactory.create_layer(
            encoder_name,
            input_size=self.input_size,
            **encoder_params,
        )

        self.classifier = LayerFactory.create_layer(
            classifier_name,
            in_features=self.encoder.output_size,
            output_size=classifier_output_size,
            **classifier_params,
        )

    def train_encoder(self) -> None:
        self.encoder.unfreeze()
        self.classifier.freeze()

    def train_classifier(self) -> None:
        self.classifier.unfreeze()
        self.encoder.freeze()

    def swap_encoder(self, encoder: torch.nn.Module) -> None:
        self.encoder = encoder

    def swap_classifier(self, classifier: torch.nn.Module) -> None:
        self.classifier = classifier

    def load_encoder(self, path: str) -> None:
        self.encoder.load(path)

    def load_classifier(self, path: str) -> None:
        self.classifier.load(path)

    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D401
        """Process inputs through encoder and classifier."""
        x = self.encoder(inputs)
        x = self.classifier(x)
        x = self.post(x)
        return self.output_activation(x)

__all__ = ["HQNN"]
