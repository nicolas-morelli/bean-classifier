import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import time


class Placeholder():
    def __init__(self):
        pass


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

    def forward(self, x):
        for bn, midlayer, do in zip(self.batch_norms, self.layers, self.dropout_layers):
            x = bn(x)
            x = midlayer(x)
            x = self.leaky_relu(x)
            x = do(x)

        x = self.output_layer(x)

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

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return accuracy
