from .BaseEncoder import BaseEncoder
from .ConvolutionEncoder import ConvolutionEncoder
from .QuanvolutionEncoder import QuanvolutionEncoder
from .LinearEncoder import LinearEncoder
from .PCAEncoder import PCAEncoder

ENCODER_REGISTRY = {
    "conv": ConvolutionEncoder,
    "quanv": QuanvolutionEncoder,
    "linear": LinearEncoder,
    "pca": PCAEncoder,
}

__all__ = [
    "BaseEncoder",
    "ConvolutionEncoder",
    "QuanvolutionEncoder",
    "LinearEncoder",
    "PCAEncoder",
    "ENCODER_REGISTRY",
]
