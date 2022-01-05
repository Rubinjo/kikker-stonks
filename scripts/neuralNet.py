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
        # self.conv = nn.Conv1d(7, 17, 2)
        self.conv = nn.Conv2d(1, 3, (7, 1))
        self.fc1 = nn.Linear(3*1*15, 32)
        self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 3)
        # self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x, hidden):
        # print(x.shape) x -> torch.Size([50, 7, 32]) => 50(batch) 7(tickers) 32(timesteps)
        out, hidden = self.lstm(x, hidden)
        # print(out.shape) out -> torch.Size([50, 7, 16]) => 50(batch) 7(tickers) 16(hidden dim)
        out = torch.unsqueeze(out, dim=1)
        out = F.relu(self.conv(out))
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = self.fc5(out)
        # We extract the scores for the final hidden state
        # out = out[:, -1]
        return out, hidden
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))