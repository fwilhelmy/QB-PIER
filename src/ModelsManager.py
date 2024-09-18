import logging
from pytorch_lightning import Trainer
from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together
from Configuration import TENSORBOARD_DIR
from pytorch_lightning.loggers import TensorBoardLogger
from art.estimators.classification import PyTorchClassifier
from models.ModelConfiguration import ModelConfiguration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelsManager:
    def __init__(self, configuration: ModelConfiguration):
        self.models = []
        self.max_epochs = configuration.max_epochs

        # TODO Refactor those things somewhere else eventually
        self.input_shape = (1, 28, 28)
        self.nb_classes = 10
    
    def register_model(self, model, configuration):
        # Store the model, trainer, and classifier
        self.models.append({'model': model, 'configuration': configuration})
        logging.info(f"Model '{model.model_name}' registered.")
        return self.models

    def prepare_models(self, train_dataset, val_dataset, test_dataset=None):
        training_results = {}
        evaluation_results = {}
        
        for model in self.models:
            model_name = model['name']
            model_instance = model['model']

            # TODO self.logger_rootdir in LogsManager now, should call the get_logger
            logger = TensorBoardLogger(self.logger_rootdir, name="logs_test_" + model_name)
            trainer = Trainer(logger=logger, max_epochs=self.max_epochs, default_root_dir=self.logger_rootdir)

            # Train the model
            logging.info(f"Training model '{model_name}'...")
            trainer.fit(model_instance, train_dataset, val_dataset)
            training_results[model_name] = trainer.logged_metrics
            
            # Test the model
            if test_dataset is None: continue
            logging.info(f"Testing model '{model_name}'...")
            test_results = trainer.test(model_instance, dataloaders=test_dataset)[0]
            evaluation_results[model_name] = test_results
        
        return training_results, evaluation_results