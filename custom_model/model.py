import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example loader

def load_custom_model(weights_path=None, device='cpu'):
    model = CustomModel()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Example inference function

def generate_text(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    # Convert output tensor to text (placeholder)
    return str(output.cpu().numpy())
