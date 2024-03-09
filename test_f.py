import torch
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

if __name__ == "__main__":
    with open('training_data/testing_data.txt', 'r') as file:
        sz = int(file.readline())
        input = torch.zeros([sz, 2], dtype=torch.float32)
        output = torch.zeros([sz, 1], dtype=torch.float32)
        i = 0
        for line in file:
            # Process each line as needed
            data = line.strip().split()
            input[i][0] = float(data[0])
            input[i][1] = float(data[1])
            output[i][0] = float(data[2])
            i += 1

    weights_path = "training_data/physics.pth"
    net = FeedForwardNN(2, 256, 1)
    net.load_state_dict(torch.load(weights_path))
    net.eval()
    outputs = net(input)
    err = outputs - output
    print(err)

