debug = True

# Constants for qunodes configuration
DEVICE = "lightning.qubit"
INTERFACE = "torch"
DIFF_METHOD = "parameter-shift"

DATA_DIR = "data/processed"

# TODO add the run_nisq flag for the CLI (default to False): if True, the model will be run on a NISQ device
# TODO Add the fast_dev_run flag for the CLI (default to False): Can be true to run 5 batches or a specific number