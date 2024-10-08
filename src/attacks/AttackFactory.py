from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# Mapping of attack names to their corresponding ART attack classes
SUPPORTED_ATTACKS = {
    "FGSM": FastGradientMethod,
    "PGD": ProjectedGradientDescent,
}

def AttackFactory(attack_name, epsilon, **kwargs):
    if attack_name not in SUPPORTED_ATTACKS:
        raise ValueError(f"Attack '{attack_name}' is not implemented. Choose from: {list(SUPPORTED_ATTACKS.keys())}")

    attack_class = SUPPORTED_ATTACKS[attack_name]
    attack = attack_class(estimator=classifier, eps=epsilon, **kwargs)
    return attack

# TODO Work on this part
def CreateFGSMAttack(classifier, epsilon):
    return AttackFactory("FGSM", classifier, epsilon)