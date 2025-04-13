# train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from ecg_utils import load_metadata, get_top_labels, extract_labels, preprocess_signals
from config import INPUT_LENGTH, NUM_CLASSES
import numpy as np

# ------------------------------
# 1. Dataset
df = load_metadata()
top_labels = get_top_labels(df, NUM_CLASSES)
df = extract_labels(df, top_labels)
X, Y = preprocess_signals(df, top_labels)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
Y = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X, Y)

# ------------------------------
# 2. Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ------------------------------
# 3. Modelo CNN
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5),   # (batch, 32, L-4)
            nn.ReLU(),
            nn.MaxPool1d(2),       # (batch, 32, (L-4)//2)
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
            nn.Sigmoid()  # Multilabel
        )

    def forward(self, x):
        return self.net(x)

model = ECG_CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------------
# 4. Entrenamiento
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------
# 5. Guardar modelo
torch.save(model.state_dict(), "ecg_cnn_model.pt")
print("✅ Modelo guardado como ecg_cnn_model.pt")
