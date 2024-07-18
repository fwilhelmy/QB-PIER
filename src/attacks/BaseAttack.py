from typing import Any, Dict

class BaseAttack(object):
    def __init__(self, name: str, model: Any, configuration: Any) -> None:
        """
        Initialize the BaseAttack.

        :param name: The name of the attack.
        :param model: The model to attack.
        :param configuration: Configuration settings for the attack.
        """
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device

        self._attack_mode = configuration.attack_mode
        self._targeted = configuration.targeted

    @property
    def attack_mode(self) -> str:
        """Get the attack mode."""
        return self._attack_mode

    @property
    def targeted(self) -> bool:
        """Check if the attack is targeted."""
        return self._targeted

    def forward(self, *input: Any) -> Any:
        """
        Perform the attack. This method should be overridden by subclasses.

        :param input: Input to the attack.
        :raises NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError("The forward method must be overridden by subclasses")

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the configuration of the attack.

        :return: A dictionary containing the attack mode and targeted status.
        """
        return {"attack_mode": self._attack_mode, "targeted": self._targeted}
