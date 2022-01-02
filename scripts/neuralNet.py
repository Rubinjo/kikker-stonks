import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_NN(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Input: 7x1x32
        # Output: 1x1x30
        # self.cnn = nn.Conv2d(7, 1, (1, 3))
        # Input: 1x1x30
        # Output: 1x1x15
        # self.pool1 = nn.MaxPool2d(1, 2)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        # We extract the scores for the final hidden state
        out = out[:, -1]
        return out, hidden
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))