import torch.optim as optim

class OptimiserFactory:
    """Factory for creating optimizers."""

    @staticmethod
    def create(name: str, parameters, *, lr: float, weight_decay: float, momentum: float = 0.9):
        """Return an optimiser instance based on ``name``.

        Parameters
        ----------
        name : str
            Name of the optimiser. Supported values are ``adamw``, ``adam``,
            ``sgd`` and ``momentum``.
        parameters : iterable
            Parameters to optimise.
        lr : float
            Learning rate for the optimiser.
        weight_decay : float
            Weight decay coefficient.
        momentum : float, optional
            Momentum value used by some optimisers, by default ``0.9``.
        """
        name = name.lower()
        if name == "adamw":
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        if name == "momentum":
            return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {name}")
