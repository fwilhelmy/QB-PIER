from .ConvolutionEncoder import ConvolutionEncoder
from .QuanvolutionEncoder import QuanvolutionEncoder
from .MLPEncoder import MLPEncoder
from .PCAEncoder import PCAEncoder

ENCODER_REGISTRY = {
    "conv": ConvolutionEncoder,
    "quanv": QuanvolutionEncoder,
    "linear": MLPEncoder,
    "pca": PCAEncoder,
}

__all__ = [
    "ConvolutionEncoder",
    "QuanvolutionEncoder",
    "MLPEncoder",
    "PCAEncoder",
    "ENCODER_REGISTRY",
]
