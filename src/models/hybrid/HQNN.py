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
from qdevice import normalize_quantum_config


@ModelFactory.model("hqnn")
class HQNN(BaseModel):
    """Hybrid Quantum Neural Network (HQNN)."""

    def __init__(
        self,
        encoder: Mapping[str, Any] | DictConfig,
        classifier: Mapping[str, Any] | DictConfig,
        quantum_config=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.quantum_config = normalize_quantum_config(quantum_config)
        self.save_hyperparameters("encoder", "classifier", "quantum_config")

        encoder_name = encoder["name"]
        encoder_params = encoder.get("params", {}) or {}

        self.encoder = LayerFactory.create_layer(
            encoder_name,
            input_size=self.input_size,
            **encoder_params,
            quantum_config=self.quantum_config.as_dict(),
        )

        classifier_name = classifier["name"]
        classifier_params = classifier.get("params", {}) or {}

        self.classifier = LayerFactory.create_layer(
            classifier_name,
            input_size=self.encoder.output_size,
            output_size=self.output_size,
            **classifier_params,
            quantum_config=self.quantum_config.as_dict(),
        )

        self.output_activation = LogSoftmax(dim=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Process inputs through encoder and classifier."""
        x = self.encoder(inputs)
        x = self.classifier(x)
        return self.output_activation(x)
