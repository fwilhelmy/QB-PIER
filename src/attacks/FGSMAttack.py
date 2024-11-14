from attacks.BaseAttack import BaseAttack
from art.attacks.evasion import FastGradientMethod
import torch
from models.BaseModel import BaseModel

class FGSMAttack(BaseAttack):
    epsilon: float

    def __init__(self, epsilon: float):
        super().__init__("FGSM({})".format(epsilon))
        self.epsilon = epsilon

    def generate_attack(self, model, data):
        attack = FastGradientMethod(estimator=model, eps=self.epsilon)
        return attack.generate(x=data)
    
    def _fgsm(self, image, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + self.epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    # TODO Really need to rework this function before implementing PGD attack
    def evaluate_attack(self, model: BaseModel, dataset):
        adv_examples = []

        model.eval()  # Set the model in evaluation mode
        for data, target in dataset:
            # Send the data and label to the device
            tr_image = torch.tensor(data, dtype=torch.float32, requires_grad=True)
            model.requires_grad=True

            # Forward pass the data through the model
            output = model(tr_image.unsqueeze(0))
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if init_pred.item() != target:
                continue

            # Calculate the loss
            loss = model.loss_function(output, torch.tensor([target], dtype=torch.long))

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect ``datagrad``
            data_grad = tr_image.grad.data

            # Call FGSM Attack
            perturbed_data = self._fgsm(data, data_grad)
            adv_examples.append(perturbed_data)
            
        return adv_examples
