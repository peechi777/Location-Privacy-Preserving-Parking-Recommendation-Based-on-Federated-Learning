import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from functions import read_data, get_dataloader, get_model_optim

# LSTM模型定義
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
       
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

# 本地訓練
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 聚合
def average_weights(weights):
    average_weight = copy.deepcopy(weights[0])
    for key in average_weight.keys():
        for i in range(1, len(weights)):
            average_weight[key] += weights[i][key]
        average_weight[key] /= len(weights)
    return average_weight

#訓練聚合
def federated_learning_cycle(models, optimizers, train_loaders, epochs, device):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        local_weights = []
        for park, model in models.items():
            loss = train_model(model, train_loaders[park], optimizers[park], criterion, device)
            local_weights.append(copy.deepcopy(model.state_dict()))
            print(f"Training loss for {park} in epoch {epoch}: {loss}")
        global_weights = average_weights(local_weights)
        for park, model in models.items():
            model.load_state_dict(global_weights)

# 數據處理
train_data, valid_data, test_data = read_data()
dataset_dict, data_loader_dict = get_dataloader(train_data, valid_data, test_data, batch_size=32, lookback=10, predict_length=1)


model_params = {'input_dim': 1, 'hidden_dim': 50, 'num_layers': 1, 'output_dim': 1}
models, optimizers = get_model_optim(train_data.columns, 'lstm', **model_params)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for model in models.values():
    model.to(device)

federated_learning_cycle(models, optimizers, data_loader_dict['train'], epochs=50, device=device)
