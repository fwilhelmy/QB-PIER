import torch

debug = True

DATA_DIR = "./data/processed"
LOGS_DIR = "./logs"

CIRCUIT_STYLE = "solarized_light"

# Constants for qunodes configuration
# default.qubit lightning.qubit
# can also add monarq or ibm quebec etc.. this will run the experiments on nisq devices
QDEVICE = "default.qubit"
INTERFACE = "torch"
QDIFF = "backprop"