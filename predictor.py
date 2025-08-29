import torch
import torch.nn as nn
import numpy as np

class DeepPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict(data, model=None):
    """
    Predict output for given data using a deep learning model.
    data: list or np.array of shape (n_samples, n_features)
    model: DeepPredictor instance (if None, a random one is created)
    """
    if model is None:
        input_dim = data.shape[1] if hasattr(data, 'shape') else len(data[0])
        model = DeepPredictor(input_dim)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        output = model(x)
    return output.numpy().tolist()
