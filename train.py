import json
import torch
import numpy as np
import random
import nltk
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, bag_of_words
from chatbot_model import NeuralNet

with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!", ","]
all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

# Dataset prep
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    label = tags.index(tag)
    X_train.append(bag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __len__(self):
        return len(X_train)
    def __getitem__(self, idx):
        return X_train[idx], y_train[idx]

# Hyperparams
batch_size = 8
input_size = len(X_train[0])
hidden_size = 128  # Increased for richer learning
output_size = len(tags)
learning_rate = 0.001
num_epochs = 500  # More epochs for better convergence

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")

for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

print("Training complete. Saving model...")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "model.pth")
print("Saved as model.pth")

