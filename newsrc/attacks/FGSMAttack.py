from attacks.BaseAttack import BaseAttack
import torch
from tqdm import tqdm
import copy

class FGSMAttack(BaseAttack):
    def __init__(self, dataset):
        super().__init__("FGSM")
        self.dataset = dataset
        self.gradients = {}

    def _fgsm(self, image, data_grad, epsilon):
        if epsilon == 0:
            return image
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
        
    def fgsm2(self, model, X, y, epsilon):
        """ Construct FGSM adversarial examples on the examples X"""
        delta = torch.zeros_like(X.to(torch.float32), requires_grad=True)
        preds = model(X + delta)
        loss = model.loss_function(preds, y)
        loss.backward()
        return X + epsilon * delta.grad.detach().sign()
    
    # Need to put the model in evaluation mode before calling this function
    def _calculate_gradient(self, model, data, target):
        # Send the data and label to the device
        tr_image = data.clone().detach().to(torch.float32).requires_grad_(True)
        # model.requires_grad=True

        # Forward pass the data through the model
        output = model(tr_image.unsqueeze(0))

        # Calculate the loss
        loss = model.loss_fn(output, torch.tensor([target], dtype=torch.long))

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = tr_image.grad

        return data_grad

    def attack(self, model, epsilon):
        if model.model_name not in self.gradients.keys():
            self.gradients[model.model_name] = []
            model.eval()  # Set the model in evaluation mode

            n_element = 0
            tmp_state = copy.deepcopy(model.state_dict())
            for data, target in tqdm(self.dataset, desc=f"Computing gradients for {model.model_name}"):
                self.gradients[model.model_name].append(self._calculate_gradient(model, data, target))
                model.load_state_dict(tmp_state)
                if n_element > 1750:
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

        return torch.stack(adv_exemples)

        #adv_exemples = self.fgsm2(model, self.dataset.data[:150].unsqueeze(1), self.dataset.targets[:150], epsilon)
        # TODO Check autograd
        # TODO Learning Rate Scheduling
            # gamma