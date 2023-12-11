import torch.nn as nn
import torch.nn.init as init
import control

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, control_type):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.fc3 = nn.Linear(hidden_size, 2)
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc3.bias)
        if control_type == control.Control_Type.BASELINE:
            self.output_activation = nn.Sigmoid()
        elif control_type == control.Control_Type.PID or control_type == control.Control_Type.EP:
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Sigmoid()

        # self.fc4 = nn.Linear(2, output_size)
        # init.xavier_uniform_(self.fc4.weight)
        # init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.output_activation(x)
        x = self.relu(x)

        # x = self.fc4(x)
        # x= self.output_activation(x)
        return x