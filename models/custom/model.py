import torch
import torch.nn as nn


class CustomModel(nn.Module):
    """Simple feed-forward network with dropout for better generalization."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
