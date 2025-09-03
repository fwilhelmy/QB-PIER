- Interactive figures
- Refactoring models
  - Improve HQNN so it's more easy to configure in more details
  - Move DimVQC and only Quantum models to `src/models/quantum`
  - All hybrid models should work and ready to start training
- Refactoring the main scripts
  - train.py
  - attack.py
  - loading.py (with refactored loading function)
- Starting to document
  - All the models type
  - The main scripts
  - The datasets
  - The attacks


ERROR: 
- the name of the models when running the attacks is not the name of the top file but the old saved name, need to change that...
- LoggingManager should have a function to log errors, it's simpler
  - two spots right now
    - in BaseModel fit
    - in MetricsContainer load 