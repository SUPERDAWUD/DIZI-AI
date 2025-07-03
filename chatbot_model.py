import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.drop1 = nn.Dropout(p=0.2)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.drop1(out)
        out = F.relu(self.l2(out))
        out = self.drop2(out)
        return self.l3(out)

