import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder


# Supported models
import brand
from sklearn.ensemble import RandomForestClassifier


# TODO: Cambiar a que tome el CSV directo, asi puedo sacar el test_df afuera
def objective(trial, data_frame, model, id, target, batch_size=64, lstart=2, lend=50, wstart=10, wend=500):

    if issubclass(model, brand.Placeholder):
        en = OneHotEncoder(sparse_output=False)
        enco = en.fit_transform(pd.DataFrame(data_frame[target]))
        enc = pd.DataFrame(enco, columns=en.get_feature_names_out()).fillna(0)
        data_frame = pd.concat([data_frame.drop(target, axis=1), enc], axis=1)

        target = enc.columns

    # pos_weight = compute_class_weight(class_weight="balanced", classes=np.unique(data_frame[target]), y=data_frame[target])

    train_df, val_df = train_test_split(data_frame, test_size=0.2, random_state=47)

    if issubclass(model, BaseEstimator):
        if issubclass(model, RandomForestClassifier):
            hypers = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=2),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=1)
            }

            model_instance = model(**hypers)
            model_instance.fit(train_df.drop(target, axis=1), train_df[target])

        score = model_instance.score(val_df.drop(target, axis=1), val_df[target])
        accuracy = score
        print(f'RF Accuracy: {accuracy}')

    elif issubclass(model, brand.Placeholder):

        train_dataset = brand.CSVDataset(data_frame=train_df, idname=id, targetname=target)
        test_dataset = brand.CSVDataset(data_frame=val_df, idname=id, targetname=target)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nlength = trial.suggest_int('nlength', lstart, lend, step=2)
        nwidth = trial.suggest_int('nwidth', wstart, wend, step=5)  # train_dataset.shape()[0], multiplier * train_dataset.shape()[0], step=5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.6, step=0.05)

        model = brand.ModelBuilder(train_dataset.shape()[0], len(target), nlength, nwidth, dropout_rate).to(device).to(torch.float64).to(device)

        loss_fn = nn.CrossEntropyLoss()  # weight=torch.tensor(pos_weight).to(device))

        optimizer = optim.NAdam(model.parameters(),
                                lr=trial.suggest_float('lr', 0.001, 0.01, step=0.0005),
                                betas=(trial.suggest_float('beta_1', 0.79, 0.99, step=0.05), trial.suggest_float('beta_2', 0.99, 0.999, step=0.001)),
                                eps=trial.suggest_float('eps', 1e-8, 1e-6, step=1e-9))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=5)

        early_stopper = brand.EarlyStopper(patience=5, min_delta=0)

        for epoch in range(50):
            train_loss = brand.train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
            validation_loss, validation_accuracy = brand.validate_one_epoch(model, train_dataloader, loss_fn, device)

            # print(f'{epoch}. ({nlength},{nwidth}) Val Accuracy: {validation_accuracy}')

            if early_stopper.early_stop(validation_loss):
                break

            scheduler.step(validation_loss)

            accuracy = brand.evaluate(model, test_dataloader, loss_fn, device)

        print(f'NN Accuracy: {accuracy}')

    return accuracy
