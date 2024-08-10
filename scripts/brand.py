import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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
        self.data_frame = data_frame.drop([idname, targetname], axis=1)
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
    correct_predictions = 0
    total_predictions = 0
    running_batch = 0.0

    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            labels = labels.float()
            loss = loss_fn(outputs, labels)  

            running_loss += loss.item()
            running_batch += len(labels)

            predicted = (outputs >= threshhold).float()
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            running_loss += loss.item()
            running_batch += len(labels)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

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

            predicted = (outputs >= threshhold).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return accuracy


def objective(trial, csv, id, target, batch_size=64, lstart=2, lend=50, wstart=10, wend=3000):

    data_frame = pd.read_csv(csv)

    num_pos = data_frame[data_frame[target] == 1][id].count()
    num_neg = data_frame[data_frame[target] == 0][id].count()
    pos_weight = num_neg / num_pos

    train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_dataset = CSVDataset(data_frame=train_df, idname=id, targetname=target)
    test_dataset = CSVDataset(data_frame=val_df, idname=id, targetname=target)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nlength = trial.suggest_int('nlength', lstart, lend, step=2)
    nwidth = trial.suggest_int('nwidth', wstart, wend, step=5)  # train_dataset.shape()[0], multiplier * train_dataset.shape()[0], step=5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.6, step=0.05)

    print('Modelo en preparacion...', end=' ')
    model = ModelBuilder(train_dataset.shape()[0], train_dataset.shape()[1], nlength, nwidth, dropout_rate).to(device).to(torch.float64)
    print('Listo!')

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    optimizer = optim.NAdam(model.parameters(),
                            lr=trial.suggest_float('lr', 0.001, 0.01, step=0.0005),
                            betas=(trial.suggest_float('beta_1', 0.79, 0.99, step=0.05), trial.suggest_float('beta_2', 0.99, 0.999, step=0.001)),
                            eps=trial.suggest_float('eps', 1e-8, 1e-6, step=1e-9))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5)

    early_stopper = EarlyStopper(patience=15, min_delta=0)

    for epoch in range(150):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        validation_loss, validation_accuracy = validate_one_epoch(model, train_dataloader, loss_fn, device)

        print(f'{epoch}. ({nlength},{nwidth}) Val Accuracy: {validation_accuracy}')

        if early_stopper.early_stop(validation_loss):
            break

        scheduler.step(validation_loss)

    accuracy = evaluate(model, test_dataloader, loss_fn, device)

    return accuracy
