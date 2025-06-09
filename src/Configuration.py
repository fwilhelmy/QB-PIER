import torch
# TODO MOVE TO ARGPARSE

debug = True

# Constants for qunodes configuration
# default.qubit lightning.qubit
# can also add monarq or ibm quebec etc.. this will run the experiments on nisq devices
QDEVICE = "default.qubit"
INTERFACE = "torch"
QDIFF = "backprop"

DATA_DIR = "data/processed"

SAVES_DIR = "saves"
LOGS_DIR = "logs"

CPU_NUMBER = 1

device: str = "cuda" if torch.cuda.is_available() else "cpu"
seed: int = 42    
verbose: bool = True
log_dir: str = 'logs'
ghost_run: bool = False # implement my ghost run code