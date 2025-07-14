import torch.optim as optim

def OptimiserFactory(optimiser_name: str, parameters, learning_rate: float, **optimiser_config):
    """Return an optimiser instance based on ``optimiser_name``.

    Parameters
    ----------
    optimiser_name : str
        Name of the optimiser. Supported values are ``adamw``, ``adam``,
        ``sgd`` and ``momentum``.
    parameters : iterable
        Parameters to optimise.
    learning_rate : float
        Learning rate for the optimiser.
    optimiser_config : dict
        Extra keyword arguments forwarded to the optimiser constructor
        (e.g. ``momentum`` or ``weight_decay``).
    """

    name = optimiser_name.lower()
    if name == "adamw":
        return optim.AdamW(parameters, lr=learning_rate, **optimiser_config)
    if name == "adam":
        return optim.Adam(parameters, lr=learning_rate, **optimiser_config)
    if name in {"sgd", "momentum"}:
        return optim.SGD(parameters, lr=learning_rate, **optimiser_config)
    raise ValueError(f"Unknown optimizer: {optimiser_name}")
