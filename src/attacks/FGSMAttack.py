from attacks.BaseAttack import BaseAttack
import torch
from models.BaseModel import BaseModel

class FGSMAttack(BaseAttack):
    def __init__(self, dataset):
        super().__init__("FGSM")
        self.dataset = dataset
        self.gradients = {}

    def _fgsm(self, image, data_grad, epsilon):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
        
    def _calculate_gradient(self, model, data, target):
        # Send the data and label to the device
        tr_image = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        model.requires_grad=True

        # Forward pass the data through the model
        output = model(tr_image.unsqueeze(0))
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target:
            return

        # Calculate the loss
        loss = model.loss_function(output, torch.tensor([target], dtype=torch.long))

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = tr_image.grad.data

        return data_grad
    
    def calculate_gradient(self, model):
        self.gradients[model.model_name] = []
        model.eval()  # Set the model in evaluation mode
        n_entity = 0
        for data, target in self.dataset:
            gradient = self._calculate_gradient(model, data, target)
            if gradient is None:
                continue
            self.gradients[model.model_name].append(gradient)

            # TODO Remove that
            if n_entity > 150:
                break
            n_entity += 1

    def attack(self, model, epsilon):
        if model.model_name not in self.gradients.keys():
            self.calculate_gradient(model)

        n_entity = 0
        adv_exemples = []
        for data, grad in zip(self.dataset, self.gradients[model.model_name]):
            if grad is None:
                print("oopsie")
                continue
            adv_exemples.append(self._fgsm(data[0], grad, epsilon))
            # TODO Remove that
            if n_entity > 150:
                break
            n_entity += 1
        
        return adv_exemples

    # def attack(self, model, data, target):
    #     # Send the data and label to the device
    #     tr_image = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    #     model.requires_grad=True

    #     # Forward pass the data through the model
    #     output = model(tr_image.unsqueeze(0))
    #     init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    #     # If the initial prediction is wrong, don't bother attacking, just move on
    #     if init_pred.item() != target:
    #         return

    #     # Calculate the loss
    #     loss = model.loss_function(output, torch.tensor([target], dtype=torch.long))

    #     # Zero all existing gradients
    #     model.zero_grad()

    #     # Calculate gradients of model in backward pass
    #     loss.backward()

    #     # Collect ``datagrad``
    #     data_grad = tr_image.grad.data

    #     # Call FGSM Attack
    #     return self._fgsm(data, data_grad)

    # # TODO Really need to rework this function before implementing PGD attack
    # def evaluate_attack(self, model: BaseModel, dataset):
    #     adv_examples = []

    #     model.eval()  # Set the model in evaluation mode
    #     n_entity = 0
    #     for data, target in dataset:
    #         perturbed_data = self.attack(model, data, target)
    #         if perturbed_data is not None:
    #             adv_examples.append(perturbed_data)
    #         # TODO Remove that
    #         if n_entity > 30:
    #             break
    #         n_entity += 1
            
    #     return adv_examples