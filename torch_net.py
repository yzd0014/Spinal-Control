import torch.nn as nn
import torch.nn.init as init

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc3.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x