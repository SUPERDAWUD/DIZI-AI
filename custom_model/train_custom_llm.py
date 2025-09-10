import os
import json
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import pytesseract

from model import CustomModel

def load_training_data(data_dir: str = "user_chats") -> List[Tuple[str, str]]:
    """Load training pairs from JSON chats, code files, or images.

    If the directory does not exist, it is created and a FileNotFoundError is
    raised to prompt the user to add data.
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(
            f"[load_training_data] Created missing directory '{data_dir}'. "
            "Add training files and rerun training."
        )
        return []

    data: List[Tuple[str, str]] = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                chats = json.load(f)
            for chat_entry in chats:
                for msg in chat_entry.get("chat", []):
                    if msg["user"] != "DIZI":
                        input_text = msg["message"]
                    else:
                        target_text = msg["message"]
                        data.append((input_text, target_text))
        elif fname.endswith(".py"):
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            data.append((code, code))
        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(fpath)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    data.append((text, text))
            except Exception as e:
                print(f"[load_training_data] Failed to process image {fname}: {e}")
    return data

def text_to_tensor(text, vocab, max_len: int = 32):
    """Encode text to a fixed-size tensor using a simple char-index mapping."""
    indices = [vocab.get(c, 0) for c in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.float32)

def build_vocab(data):
    chars = set()
    for inp, tgt in data:
        chars.update(inp)
        chars.update(tgt)
    vocab = {c: i+1 for i, c in enumerate(sorted(chars))}
    return vocab

def train():
    data = load_training_data()
    if not data:
        print('No training data found!')
        return
    vocab = build_vocab(data)
    model = CustomModel(input_dim=32, hidden_dim=128, output_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for inp, tgt in data:
            inp_tensor = text_to_tensor(inp, vocab)
            tgt_tensor = text_to_tensor(tgt, vocab)
            optimizer.zero_grad()
            output = model(inp_tensor)
            loss = criterion(output, tgt_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}')
    torch.save(model.state_dict(), os.path.join('custom_model', 'custom_llm_weights.pth'))
    print('Training complete. Weights saved to custom_llm_weights.pth')

if __name__ == '__main__':
    train()
