# Bean Classifier

This repository contains a full, end to end, Machine Learning project. It uses the Dry Beans dataset (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). 

By masking any model implementation with the Scikit Learn interface, it is able to seamlessly test hyperparameters and choose the best model (on accuracy). Moreover, it selects the best feature selection method (in a production environment it might not be useful, it is done in an educational spirit).

## Scripts

### Feature Selection

The script feature-selection.py is the first step of the process. It creates two different datasets with dimensionality reduction, one with PCA, one with SelectFromModel.

### Model Selection

The next script in the chain is model-selection.py. It optimizes hyperparameters for all the model selected and then chooses the best performing one, testing it against the test dataset.

## Brand

Brand is a Neural Network written in Pytorch by myself to understand the library and the common NN steps and components. It was written previously in Tensorflow and translated into Pytorch, before the development of this project. It was tweaked to accept multiclass prediction and the interface Brand() was developed to facilitate the scripts flow.

## Improvements

- FocalLoss: coding a FocalLoss class might help with the class imbalance versus using the class weights.
- Wrapper: wraping Scikit and other models in a class that holds the hyperparameter values might simplify the objective function.
- Feature engineering: modifying the features might improve metrics, it was not done as the focus of the project was on the models, not the analysis.