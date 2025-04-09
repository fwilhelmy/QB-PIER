from attacks.BaseAttack import BaseAttack
import torch
from tqdm import tqdm

# simple implementation of projected gradient descent (PGD) attack (without randomized starting points — cf. BIM)
# for an introduction to PGD, see https://adversarial-ml-tutorial.org/adversarial_examples/#projected-gradient-descent

# PGD code taken from: https://pennylane.ai/qml/demos/tutorial_adversarial_attacks_QML
class PGDAttack(BaseAttack):
    def __init__(self, dataset):
        super().__init__("PGD")
        self.dataset = dataset

    def PGD(self, model, feats, labels, epsilon=0.1, alpha=0.01, num_iter=10):
        # initialize image perturbations with zero
        delta = torch.zeros_like(feats.to(torch.float), requires_grad=True)
        for t in tqdm(range(num_iter)):
            feats_adv = feats + delta
            outputs = [model(f) for f in feats_adv]

            # forward & backward pass through the model, accumulating gradients
            l = model.loss_function(torch.stack(outputs), labels)
            l.backward()

            # use gradients with respect to inputs and clip the results to lie within the epsilon boundary
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()
    
    def attack(self, model, epsilon):
        feats = self.dataset.data[:150]
        labels = self.dataset.targets[:150]
        return self.PGD(model, feats, labels, epsilon=epsilon)