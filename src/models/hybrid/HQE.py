"""Hybrid quantum encoder model using the HQNN architecture."""

from models import ModelFactory
from .HQNN import HQNN


@ModelFactory.model("hqe")
class HQE(HQNN):
    """Hybrid Quantum Encoder built upon :class:`HQNN`."""

    def __init__(
        self,
        hidden_sizes,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        **kwargs,
    ) -> None:
        output_size = kwargs.get("output_size")
        encoder_params = {"hidden_sizes": hidden_sizes}
        classifier_params = {
            "in_qlayer": output_size,
            "n_qdepth": n_qdepth,
            "ansatz": ansatz,
            "embedding": embedding,
        }
        super().__init__(
            encoder="linear",
            classifier="vqc",
            encoder_params=encoder_params,
            classifier_params=classifier_params,
            **kwargs,
        )
        self.save_hyperparameters(
            "hidden_sizes",
            "n_qdepth",
            "ansatz",
            "embedding",
        )
