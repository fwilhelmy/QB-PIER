import os
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from torch import load, save
import json

class LoggingService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, root_dir=None):
        if not hasattr(self, 'initialized'):  # Ensure the initialization code runs only once
            self.initialized = True
            
            # Create a directory to store the results
            timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
            if not root_dir: root_dir = "results"
            self._root_dir = f"{root_dir}/{timestring}"
            os.makedirs(self._root_dir, exist_ok=True)
            print(f"Tensorboard path: {self._root_dir}/tensorboard")

    def get_root_dir(self):
        """Return the root directory of the logs."""
        return self._root_dir
    
    def save_figure(self, figure, title, model=None):
        figure_dir = f"{self._root_dir}/" + (model.model_name if model else "")
        figure.savefig(f"{figure_dir}/{title}.png")

    # TODO should not be in the LOGS, instead create a new folder called models and we save there
    # during initilisation you check if this model has already been saved and load it if it's the case
    # hierarchy : saves -> Dataset -> ModelClass -> 
    # HighSpecs (Overview specs : Epochs, Learning Rate, etc..) ->
    # LowSpecs (All specs that are configurable) -> model_state.pt

    # Also should save the training figure and a log of the training in json to see how the training went
    # In a dataset when it's loaded for the first time, should save dataset figures and store into
    def save_model(self, model):
        # TODO instead use torch.save
        model_config = f"Model name:{model.model_name} \n{str(model)}"
        self.save_file(model_config, "configuration.txt", model)
        save(model.state_dict(), f"{self._root_dir}/{model.model_name}/model_state.pt")

    def save_file(self, file_content, file_name, model=None):
        file_dir = f"{self._root_dir}/" + (model.model_name if model else "")
        with open(f"{file_dir}/{file_name}", "w") as file:
            file.write(file_content)

    def save_metrics(self, model):
        metrics_dir = f"{self._root_dir}/{model.model_name}"
        with open(f"{metrics_dir}/metrics.json", "w") as f:
            json.dump(model.all_metrics, f, indent=4)