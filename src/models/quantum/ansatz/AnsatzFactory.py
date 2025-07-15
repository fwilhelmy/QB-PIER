from __future__ import annotations

from typing import Type


from .BaseAnsatz import BaseAnsatz
from . import (
    BaseAnsatz,
    BasicEntanglingAnsatz,
    IdentityAnsatz,
    NoEntanglingAnsatz,
    RandomStructureAnsatz,
    ResidualAnsatzWrapper,
    ReUpAnsatzWrapper,
    StrongEntanglingAnsatz,
)

ansatz_classes: dict[str, Type[BaseAnsatz]] = {
    "BasicEntanglingAnsatz": BasicEntanglingAnsatz,
    "IdentityAnsatz": IdentityAnsatz,
    "NoEntanglingAnsatz": NoEntanglingAnsatz,
    "RandomStructureAnsatz": RandomStructureAnsatz,
    "ResidualAnsatzWrapper": ResidualAnsatzWrapper,
    "ReUpAnsatzWrapper": ReUpAnsatzWrapper,
    "StrongEntanglingAnsatz": StrongEntanglingAnsatz,
}
    

def AnsatzFactory(input_size, n_layers, embedding, ansatz):
    """
    Factory function to create an Ansatz instance based on the provided class and configuration.
    
    Args:
        input_size (int): Size of the input.
        n_layers (int): Number of layers in the ansatz.
        embedding (str): Type of embedding to use.
        ansatz_class (str): Name of the ansatz class to instantiate.
        ansatz_config (dict): Configuration parameters for the ansatz.
    Returns:
        BaseAnsatz: An instance of the specified ansatz class.
    """
    if ansatz is None: ansatz = {}
    ansatz_class = ansatz.get("class", "StrongEntanglingAnsatz")
    ansatz_config = ansatz.get("params", {})
    if ansatz_class not in ansatz_classes:
        raise ValueError(f"Unsupported ansatz class: {ansatz_class}")
    
    ansatz_cls = ansatz_classes[ansatz_class]
    return ansatz_cls(
        input_size=input_size,
        n_layers=n_layers,
        embedding=embedding,
        **ansatz_config
    )
    

