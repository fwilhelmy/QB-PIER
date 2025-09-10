from .BaseClassifier import BaseClassifier
from .ParallelVQCClassifier import ParallelVQCClassifier
from .VQCClassifier import VQCClassifier
from .MLPClassifier import MLPClassifier

CLASSIFIER_REGISTRY = {
    "parallel_vqc": ParallelVQCClassifier,
    "vqc": VQCClassifier,
    "mlp": MLPClassifier,
}

__all__ = [
    "BaseClassifier",
    "ParallelVQCClassifier",
    "VQCClassifier",
    "MLPClassifier",
    "CLASSIFIER_REGISTRY",
]
