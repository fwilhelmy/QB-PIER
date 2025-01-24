from attacks.BaseAttack import BaseAttack
import torch
from tqdm import tqdm

class FGSMAttack(BaseAttack):
    def __init__(self, dataset):
        super().__init__("FGSM")
        self.dataset = dataset
        self.gradients = {}

    def _fgsm(self, image, data_grad, epsilon):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1) * 255
        return perturbed_image
        
    # Need to put the model in evaluation mode before calling this function
    def _calculate_gradient(self, model, data, target):
        # Send the data and label to the device
        tr_image = data.clone().detach().to(torch.float32).requires_grad_(True)
        model.requires_grad=True

        # Forward pass the data through the model
        output = model(tr_image.unsqueeze(0))

        # Calculate the loss
        loss = model.loss_function(output, torch.tensor([target], dtype=torch.long))

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = tr_image.grad.data

        return data_grad

    def attack(self, model, epsilon):
        if model.model_name not in self.gradients.keys():
            self.gradients[model.model_name] = []
            model.eval()  # Set the model in evaluation mode

            n_element = 0
            for data, target in tqdm(self.dataset, desc=f"Computing gradients for {model.model_name}"):
                self.gradients[model.model_name].append(self._calculate_gradient(model, data, target))
                if n_element > 150:
                    break
                n_element += 1
                

        adv_exemples = []
        n_element = 0
        for idx, data in enumerate(self.dataset):
            image, _ = data
            gradient = self.gradients[model.model_name][idx]
            adv_exemples.append(self._fgsm(image, gradient, epsilon))
            if n_element > 150:
                break
            n_element += 1
        
        return adv_exemples