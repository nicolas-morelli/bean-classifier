import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Brand():
    def __init__(self, nlength, nwidth, dropout_rate, lr, beta_1, beta_2, eps, weights, epochs, patience, target=None, id=None, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlength = nlength
        self.nwidth = nwidth
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.betas = (beta_1, beta_2)
        self.eps = eps
        self.weights = weights
        self.epochs = epochs
        self.patience = patience
        self.target = target
        self.id = id
        self.batch_size = batch_size

    def fit(self, train_df, train_df_targets):
        train_df = pd.concat([train_df, train_df_targets], axis=1)
        train_dataset = CSVDataset(data_frame=train_df, idname=self.id, targetname=self.target)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model = ModelBuilder(train_dataset.shape()[0], len(self.target), self.nlength, self.nwidth, self.dropout_rate).to(self.device).to(torch.float64).to(self.device)

        loss_fn = nn.BCEWithLogitsLoss(torch.tensor(self.weights).to(self.device))

        optimizer = optim.NAdam(model.parameters(),
                                lr=self.lr,
                                betas=self.betas,
                                eps=self.eps)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=5)

        early_stopper = EarlyStopper(patience=self.patience, min_delta=0)

        for _ in range(self.epochs):
            _ = train_one_epoch(model, train_dataloader, optimizer, loss_fn, self.device)
            validation_loss, validation_accuracy = validate_one_epoch(model, train_dataloader, loss_fn, self.device)

            # print(f'{epoch}. ({self.nlength},{self.nwidth}) Val Accuracy: {validation_accuracy}')

            if early_stopper.early_stop(validation_loss):
                break

            scheduler.step(validation_loss)

        self.loss_fn = loss_fn
        self.model = model

        return model

    def score(self, val_df, val_df_targets):
        val_df = pd.concat([val_df, val_df_targets], axis=1)
        test_dataset = CSVDataset(data_frame=val_df, idname=self.id, targetname=self.target)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        accuracy = evaluate(self.model, test_dataloader, self.loss_fn, self.device)

        return accuracy

    def predict_proba(self, inputs):
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(torch.from_numpy(inputs.to_numpy()).to(torch.float64).to(self.device))

        return outputs


class ModelBuilder(nn.Module):
    def __init__(self, ninputs, noutputs, nlength, nwidth, dropout_rate):
        super(ModelBuilder, self).__init__()
        self.nlength = nlength
        self.nwidth = nwidth
        self.dropout_rate = dropout_rate
        self.ninputs = ninputs
        self.noutputs = noutputs

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        dim = ninputs
        for _ in range(nlength):
            self.batch_norms.append(nn.BatchNorm1d(dim))
            layer = nn.Linear(dim, nwidth)
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            self.layers.append(layer)
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            dim = nwidth

        self.output_layer = nn.Linear(dim, noutputs)
        torch.nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for bn, midlayer, do in zip(self.batch_norms, self.layers, self.dropout_layers):
            x = bn(x)
            x = midlayer(x)
            x = self.leaky_relu(x)
            x = do(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x


class CSVDataset(Dataset):
    def __init__(self, data_frame, idname, targetname):
        data_frame = data_frame.reset_index(drop=True)
        self.data_frame = data_frame.drop([idname] + targetname, axis=1) if idname is not None else data_frame.drop(targetname, axis=1)
        self.labels = pd.DataFrame(data_frame[targetname])
        self.idname = idname
        self.targetname = targetname

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data_frame.iloc[idx].to_numpy(), self.labels.iloc[idx].to_numpy()

    def shape(self):
        return (len(self.data_frame.columns), 1)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    running_batch = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.float()

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_batch += len(labels)

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


def validate_one_epoch(model, dataloader, loss_fn, device, threshhold=0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    running_batch = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.float()
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            running_batch += len(labels)

            predicted_indices = torch.argmax(outputs, dim=1)
            true_indices = torch.argmax(labels, dim=1)

            correct += (predicted_indices == true_indices).sum().item()
            total += labels.size(0)

            running_loss += loss.item()
            running_batch += len(labels)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return epoch_loss, accuracy


def evaluate(model, dataloader, loss_fn, device, threshhold=0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.float()
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            predicted_indices = torch.argmax(outputs, dim=1)
            true_indices = torch.argmax(labels, dim=1)

            correct += (predicted_indices == true_indices).sum().item()
            total += labels.size(0)

    # avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return accuracy
