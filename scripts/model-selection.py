import objective
import functools
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

# Models
import brand
from sklearn.ensemble import RandomForestClassifier


def optimize_hyperparameters(df, model):
    sampler = optuna.samplers.GPSampler(seed=47)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    obj = functools.partial(objective.objective, data_frame=df, model=model, id=None, target='target')
    study.optimize(obj, n_trials=5)
    trial = study.best_trial

    return trial.value


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sfm = pd.read_csv('data//sfm.csv')
    pca = pd.read_csv('data//pca.csv')

    train_pca, test_pca = train_test_split(pca, test_size=0.2, random_state=47)
    train_sfm, test_sfm = train_test_split(sfm, test_size=0.2, random_state=47)

    models = [brand.Placeholder, RandomForestClassifier]

    for train in [train_pca, train_sfm]:
        for model in models:
            train = train.reset_index()
            val = optimize_hyperparameters(train, model)

            print(val)

    # Elijo el mejor modelo


if __name__ == "__main__":
    main()