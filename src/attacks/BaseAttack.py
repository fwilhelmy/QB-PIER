class BaseAttack():
    attack_name: str

    def __init__(self, attack_name: str):
        self.attack_name = attack_name

    def attack(self, model, data):
        raise NotImplementedError("Attack method must be implemented...")