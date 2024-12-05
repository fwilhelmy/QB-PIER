import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import pennylane as qml

qml.about()

# https://discuss.pennylane.ai/t/question-of-parameter-shift-method-in-broadcasting/4940/6
# https://discuss.pennylane.ai/t/problems-with-gradient-computing-on-pennylane-lightning/7346/3
# https://discuss.pennylane.ai/t/question-of-parameter-shift-method-in-broadcasting/4940
# https://discuss.pennylane.ai/t/qml-training-using-shot-based-simulator/4206