"""Quanvolutional hybrid model implemented using the HQNN architecture."""

from typing import Optional

from models import ModelFactory
from .HQNN import HQNN


@ModelFactory.model("hquanv")
class HQuanv(HQNN):
    """Hybrid Quanvolutional Neural Network built on :class:`HQNN`."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 2,
        stride: Optional[int] = None,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "angle",
        rotation: str = "X",
        fc_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        stride = stride or patch_size
        encoder_params = {
            "patch_size": patch_size,
            "stride": stride,
            "n_qdepth": n_qdepth,
            "ansatz": ansatz,
            "embedding": embedding,
            "rotation": rotation,
        }
        n_patches = (img_size - patch_size) // stride + 1
        enc_output = (patch_size * patch_size) * n_patches * n_patches
        fc_hidden = fc_size or enc_output
        classifier_params = {"hidden_sizes": fc_hidden}
        super().__init__(
            encoder="quanv",
            classifier="mlp",
            encoder_params=encoder_params,
            classifier_params=classifier_params,
            **kwargs,
        )
        self.save_hyperparameters(
            "img_size",
            "patch_size",
            "stride",
            "n_qdepth",
            "ansatz",
            "embedding",
            "rotation",
            "fc_size",
        )
