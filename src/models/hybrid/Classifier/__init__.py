from .ParallelVQCClassifier import ParallelVQCClassifier
from .VQCClassifier import VQCClassifier
from .MLPClassifier import MLPClassifier

CLASSIFIER_REGISTRY = {
    "parallel_vqc": ParallelVQCClassifier,
    "vqc": VQCClassifier,
    "mlp": MLPClassifier,
}

__all__ = [
    "ParallelVQCClassifier",
    "VQCClassifier",
    "MLPClassifier",
    "CLASSIFIER_REGISTRY",
]
