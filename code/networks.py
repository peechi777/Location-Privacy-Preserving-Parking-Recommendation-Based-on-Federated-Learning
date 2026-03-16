import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os


# baselines
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_l, num_layers):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*seq_l, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, _ = self.rnn(x)  # shape [batch, seq, feature]
        x = self.flatten(x)
        x = self.linear(x)
        return x
