from src.attacks.BaseAttack import BaseAttack
from art.attacks.evasion import FastGradientMethod

class FGSMAttack(BaseAttack):
    epsilon: float

    def __init__(self, epsilon: float):
        super().__init__("FGSM({})".format(epsilon))
        self.epsilon = epsilon

    def generate_attack(self, model, data):
        attack = FastGradientMethod(estimator=model, eps=self.epsilon)
        return attack.generate(x=data)
