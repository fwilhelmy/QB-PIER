import random
import numpy as np
import torch
import torch


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(y_hat, y):
    _, predicted = torch.max(y_hat, dim=1)
    correct = (predicted == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def new_accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc


@torch.no_grad()
def evaluate_attack(self, classifier, dataset, adv_examples):
    labels = dataset.targets.cpu().numpy()
    original_preds = classifier.predict(dataset.data.unsqueeze(1).numpy())
    perturbed_preds = classifier.predict(adv_examples)

    # Calculate accuracy
    original_accuracy = calculate_accuracy(original_preds, labels)
    perturbed_accuracy = calculate_accuracy(perturbed_preds, labels)
    accuracy_difference = original_accuracy - perturbed_accuracy

    return accuracy_difference


# Calcul de la précision
def precision(y, y_gt):
    return (y.argmax(0) == y_gt.argmax(0)).mean()


# Conversion de la classe en représentation one hot
def one_hot(y, num_classes=10):
    # Crée une matrice de zéros de taille (nombre d'échantillons, nombre de classes)
    z = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    z[np.arange(y.shape[0]), y] = 1.0
    return z


def to_tensor(x):
    # Normalise les valeurs de pixel entre 0 et 1 et redimensionne l'image en un vecteur
    return x.reshape((x.shape[0], -1)).astype(np.float32) / 255.0
