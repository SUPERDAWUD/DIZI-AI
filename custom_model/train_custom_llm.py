import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from model import CustomModel

def load_training_data(data_dir='user_chats'):
    data = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.json'):
            with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
                chats = json.load(f)
            for chat_entry in chats:
                for msg in chat_entry['chat']:
                    # Example: use user message as input, bot message as target
                    if msg['user'] != 'DIZI':
                        input_text = msg['message']
                    else:
                        target_text = msg['message']
                        data.append((input_text, target_text))
    return data

def text_to_tensor(text, vocab, max_len=32):
    # Simple encoding: map chars to indices, pad/truncate
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
