from Configuration import DATA_DIR

from models.Classical.LeNet import LeNet
from models.Hybrid.HQNN_Parallel import HQNN_Parallel
from models.Hybrid.HQNN_Quanv import HQNN_Quanv


def ModelFactory(model_name: str, logdir: str = None):
    if model_name == 'LeNet': model = LeNet(model_name='LeNet_8', devices=hparams.devices, max_epochs=8, in_channels=1, classes=10)
    elif model_name == 'HQNN_Parallel': model = FMNISTManager(root=root)
    elif model_name == 'HQNN_Quanv': model = FMNISTManager(root=root)
    else: raise ValueError(f"Unsupported modle: {model_name}")

    if logdir is not None: model.l