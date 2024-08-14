import pickle
import pandas as pd
import numpy as np
import functools
import optuna
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

# Models
import brand
from sklearn.ensemble import RandomForestClassifier


def preprocess(model, data_frame, target):
    weights = None

    if issubclass(model, brand.Brand):
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(data_frame[target]), y=data_frame[target])
        en = OneHotEncoder(sparse_output=False)
        enco = en.fit_transform(pd.DataFrame(data_frame[target]))
        enc = pd.DataFrame(enco, columns=en.get_feature_names_out()).fillna(0)
        data_frame = pd.concat([data_frame.drop(target, axis=1), enc], axis=1)

        target = enc.columns

    return data_frame, target, weights


def objective(trial, data_frame, model, id, target, batch_size=64):

    data_frame, target, weights = preprocess(model, data_frame, target)

    train_df, val_df = train_test_split(data_frame, test_size=0.2, random_state=47)

    if issubclass(model, RandomForestClassifier):
        hypers = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=2),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=1)
        }

    elif issubclass(model, brand.Brand):
        hypers = {
            'nlength': trial.suggest_int('nlength', 2, 15, step=1),
            'nwidth': trial.suggest_int('nwidth', 2, 150, step=2),  # train_dataset.shape()[0], multiplier * train_dataset.shape()[0], step=5)
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.6, step=0.05),
            'lr': trial.suggest_float('lr', 0.001, 0.01, step=0.0001),
            'beta_1': trial.suggest_float('beta_1', 0.89, 0.99, step=0.01),
            'beta_2': trial.suggest_float('beta_2', 0.99, 0.9999, step=0.0001),
            'eps': trial.suggest_float('eps', 1e-8, 1e-6, step=1e-9),
            'epochs': trial.suggest_int('epochs', 100, 500, step=25),
            'patience': trial.suggest_int('patience', 2, 50, step=2),
            'weights': weights,
            'target': target,
            'batch_size': batch_size
        }

        trial.set_user_attr('weights', weights)
        trial.set_user_attr('target', target)
        trial.set_user_attr('batch_size', batch_size)

    model_instance = model(**hypers)
    model_instance.fit(train_df.drop(target, axis=1), train_df[target])

    return model_instance.score(val_df.drop(target, axis=1), val_df[target])


def optimize_hyperparameters(df, model, trials):
    sampler = optuna.samplers.TPESampler(seed=47)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    obj = functools.partial(objective, data_frame=df, model=model, id=None, target='target')
    print(f'Running {trials} trials for {model}')
    study.optimize(obj, n_trials=trials)
    trial = study.best_trial

    return trial.value, trial.params | trial.user_attrs


def main():
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    sfm = pd.read_csv('data//sfm.csv')
    pca = pd.read_csv('data//pca.csv')

    train_pca, test_pca = train_test_split(pca.sample(frac=1), test_size=0.2, random_state=47)
    train_sfm, test_sfm = train_test_split(sfm.sample(frac=1), test_size=0.2, random_state=47)

    trials = 20
    models = [brand.Brand, RandomForestClassifier]
    minval = 0

    for train, test in zip([train_sfm, train_pca], [test_sfm, test_pca]):
        for model in models:
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            val, params = optimize_hyperparameters(train, model, trials)

            print(f'Accuracy {val} for model {model} with params {params}')

            if val >= minval:
                besttrain, target, _ = preprocess(model, train, 'target')
                totest, _, _ = preprocess(model, test, 'target')
                minval = val
                bestmodel = model
                bestparams = params
                print('Latest model overtook')

    print(f'Best params {bestparams} for best model {model}')
    bestmodel = bestmodel(**bestparams)
    bestmodel.fit(besttrain.drop(target, axis=1), besttrain[target])

    print(f'Accuracy on Test Set for best model: {bestmodel.score(totest.drop(target, axis=1), totest[target])}')

    with open('data//model.pkl', 'wb') as modelfile:
        pickle.dump(bestmodel, modelfile)


if __name__ == "__main__":
    main()
