from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# Mapping of attack names to their corresponding ART attack classes
attack_mapping = {
    "FGSM": FastGradientMethod,
    "PGD": ProjectedGradientDescent,
}

def AttackFactory(attack_name, classifier, epsilon, **kwargs):
    if attack_name not in attack_mapping:
        raise ValueError(f"Attack '{attack_name}' is not implemented. Choose from: {list(attack_mapping.keys())}")

    attack_class = attack_mapping[attack_name]
    attack = attack_class(estimator=classifier, eps=epsilon, **kwargs)
    return attack

# TODO Work on this part
def CreateFGSMAttack(classifier, epsilon):
    return AttackFactory("FGSM", classifier, epsilon)