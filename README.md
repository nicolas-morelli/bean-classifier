# Tittle

This repository contains a full, end to end, Machine Learning project. It uses the Dry Beans dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).

## Scripts

### Feature Selection

The script feature-selection.py is the first step of the process. It creates two different datasets with dimensionality reduction, one with PCA, one with SelectFromModel.

### Model Selection

The next script in the chain is model-selection.py. It optimizes hyperparameters for all the model selected and then chooses the best performing one.

### Final training

The last part takes the best model and runs it and tests it against the testing set.

## Brand

Brand is a Neural Network written in Pytorch by myself to understand the library and the common NN steps and components. Its code is previous to this project.

## Lessons Learned