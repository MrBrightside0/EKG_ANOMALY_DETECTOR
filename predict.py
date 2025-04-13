# predict.py

import torch
import torch.nn as nn
import wfdb
import numpy as np
from config import INPUT_LENGTH, ECG_PATH, NUM_CLASSES
from ecg_utils import get_top_labels, load_metadata

# ------------------------------
# Modelo CNN (debe coincidir con train_cnn.py)
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_CNN()
model.load_state_dict(torch.load(r"C:\Users\cuent\Desktop\CNN HACKHATON\ecg-system\models\ecg_cnn_model.pt", map_location=device))
model.eval()

# ------------------------------
# Cargar labels
df = load_metadata()
top_labels = get_top_labels(df, NUM_CLASSES)

# ------------------------------
# Función para predecir desde .dat/.hea
def predict_ecg(record_name):
    path = f"{ECG_PATH}/{record_name}"
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, 0]  # solo una derivación

    if len(signal) < INPUT_LENGTH:
        print("🚫 Señal muy corta para evaluar.")
        return

    signal = signal[:INPUT_LENGTH]
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    with torch.no_grad():
        output = model(x)
    output = output.squeeze().numpy()

    print(f"🩺 Diagnóstico para {record_name}:")
    for label, prob in zip(top_labels, output):
        if prob > 0.5:
            print(f"✅ {label}: {prob:.2f}")
        else:
            print(f"❌ {label}: {prob:.2f}")

# ------------------------------
# Ejemplo de uso
if __name__ == "__main__":
    # Reemplaza con cualquier nombre real, como 'records500/00001_lr'
    predict_ecg("records500/00001_lr")
