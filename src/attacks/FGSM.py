import torch
import torch.nn.functional as F
from torchvision import transforms
import logging
from utils import get_device, get_lossfn
from torch.utils.data import DataLoader
import time
import numpy as np

def fgsm(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch, mean, std):
    device = get_device()
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# TODO Really need to rework this function before implementing PGD attack
def evaluate_attack(model, test_dataset, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    device = get_device()

    model.eval()  # Set the model in evaluation mode
    
    # Loop over all examples in test set
    for data, target in test_dataset:
        # Send the data and label to the device
        tr_image = torch.tensor(data, dtype=torch.float32, requires_grad=True).to(device)
        model.requires_grad=True

        # Forward pass the data through the model
        output = model(tr_image.unsqueeze(0)) # TODO Could be .argmax(1)?
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target:
            continue

        # Calculate the loss
        loss = get_lossfn()(output, torch.tensor([target], dtype=torch.long))

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = tr_image.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data_grad, mean=[0.1307], std=[0.3081])

        # Call FGSM Attack
        perturbed_data = fgsm(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target:
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_dataset))
    logging.info(('Epsilon {} - Accuracy {:.2f}').format(epsilon, final_acc))
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_dataset)} = {final_acc}\tTime taken: {time.time() - start_time}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples