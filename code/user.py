import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import copy
import numpy as np


def read_data(file_path):
    
    df = pd.read_csv(file_path)
    return df

def create_datasets(data, look_back=10):
    
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def get_dataloader(dataX, dataY, batch_size=32):
    
    tensor_x = torch.Tensor(dataX).unsqueeze(2)  
    tensor_y = torch.Tensor(dataY)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)

class FeedbackLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FeedbackLSTM, self).__init__()
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

def train_local(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
    return model.state_dict()

def apply_differential_privacy(model_weights, sensitivity, epsilon):
    
    for key in model_weights.keys():
        noise = torch.randn(model_weights[key].shape) * (sensitivity / epsilon)
        model_weights[key] += noise
    return model_weights

def federated_averaging(models: list[dict[str, torch.Tensor]]):
    averaged_model = copy.deepcopy(models[0])
    for key in averaged_model.keys():
        for i in range(1, len(models)):
            averaged_model[key] += models[i][key]
        averaged_model[key] = averaged_model[key] / len(models)
    return averaged_model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = read_data("your_data.csv")
    dataX, dataY = create_datasets(data.values)
    dataloader = get_dataloader(dataX, dataY)

    model_params = {'input_dim': 1, 'hidden_dim': 50, 'num_layers': 2, 'output_dim': 1}
    model = FeedbackLSTM(**model_params).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    #本地
    local_model = train_local(model, dataloader, optimizer, criterion, epochs=10, device=device)
    
    #差分隱私
    dp_model = apply_differential_privacy(local_model, sensitivity=1.0, epsilon=0.1)

    
    global_model = federated_averaging([dp_model])
    model.load_state_dict(global_model)

if __name__ == "__main__":
    main()
