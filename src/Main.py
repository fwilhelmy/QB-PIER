def register_model(models, model_name, model_class, config):
    """Register a model with its configuration."""
    if model_name not in models:
        models[model_name] = {
            'model_class': model_class,
            'config': config
        }
        logging.info(f"Model '{model_name}' registered.")
    else:
        logging.warning(f"Model '{model_name}' already registered.")
    return models

def train_all(models):
    """Train all registered models and return their histories."""
    results = {}
    for model_name, info in models.items():
        logging.info(f"Starting training for model: {model_name}")
        config = info['config']
        trainer = TrainingModule(config)
        model = info['model_class'](in_channels=1, classes=len(trainer.train_loader.dataset.classes)).to(config.DEVICE)
        trainer.model = model
        history = trainer.train()
        results[model_name] = history
        logging.info(f"Training completed for model: {model_name}")
    return results

def evaluate_all(models):
    """Evaluate all registered models and return their accuracies."""
    accuracies = {}
    for model_name, info in models.items():
        logging.info(f"Starting evaluation for model: {model_name}")
        config = info['config']
        trainer = TrainingModule(config)
        model = info['model_class'](in_channels=1, classes=len(trainer.train_loader.dataset.classes)).to(config.DEVICE)
        trainer.model = model
        accuracy = trainer.evaluate()
        accuracies[model_name] = accuracy
        logging.info(f"Evaluation completed for model: {model_name}. Accuracy: {accuracy:.2f}%")
    return accuracies

# Usage
models = {}
config_1 = Config(init_lr=1e-3, epochs=5)
config_2 = Config(init_lr=1e-4, epochs=10)
models = register_model(models, 'Model1', HQNN_Quanv, config_1)
models = register_model(models, 'Model2', LeNet, config_2)
training_results = train_all(models)
evaluation_results = evaluate_all(models)
