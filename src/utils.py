# import pennylane as qml
import torch
import Configuration as cfg
import pennylane as qml

# Helper functions for creating circuits
def create_qnode(n_qubits, circuit, weight_shapes, init_vals=None):
    dev = qml.device(cfg.DEVICE, wires=n_qubits)
    qnode_obj = qml.QNode(circuit, dev, interface=cfg.INTERFACE, diff_method=cfg.DIFF_METHOD)
    return qml.qnn.TorchLayer(qnode=qnode_obj, weight_shapes=weight_shapes, init_method=init_vals)

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

def evaluate_attack(self, classifier, dataset, adv_examples):
    labels = dataset.targets.cpu().numpy()
    original_preds = classifier.predict(dataset.data.unsqueeze(1).numpy())
    perturbed_preds = classifier.predict(adv_examples)

    # Calculate accuracy
    original_accuracy = calculate_accuracy(original_preds, labels)
    perturbed_accuracy = calculate_accuracy(perturbed_preds, labels)
    accuracy_difference = original_accuracy - perturbed_accuracy

    return accuracy_difference