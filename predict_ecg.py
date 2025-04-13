import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ecg_utils import load_ecg, preprocess_single_signal_torch
from models import ECGCNN  # Importa la clase ECGCNN desde el archivo correcto
from config import ECG_PATH, INPUT_LENGTH
import pandas as pd
import os

# Cargar clases desde scp_statements.csv
df_classes = pd.read_csv("scp_statements.csv")
class_map = {i: row['diagnostic'] for i, row in enumerate(df_classes['diagnostic'])}

# Cargar modelo
model = ECGCNN()
model.load_state_dict(torch.load("models/ECGCNN.pt", map_location=torch.device("cpu")))
model.eval()

# Cargar una señal (sin extensión)
filename = 'records500/00000/00001_hr'
full_path = os.path.join(ECG_PATH, filename)
signal = load_ecg(full_path)

# Preprocesar
signal_tensor = preprocess_single_signal_torch(signal)  # Debe retornar torch.tensor shape (1, 1, INPUT_LENGTH)

# Predicción
with torch.no_grad():
    output = model(signal_tensor)
    probs = F.softmax(output, dim=1).squeeze()
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()

# Mostrar resultados
print(f"Predicción: {class_map[pred_class]} (Confianza: {confidence:.2f})")

# Graficar señal y resaltar
plt.figure(figsize=(12, 4))

# Cambiar el fondo a color ECG (gris claro)
plt.gcf().set_facecolor('#f0f0f0')  # Fondo color gris claro (simula el color del ECG)

# Graficar la señal ECG
plt.plot(signal[:INPUT_LENGTH], label='ECG')

# Título con la predicción
plt.title(f"Predicción: {class_map[pred_class]} (Conf: {confidence:.2f})")

# Etiquetas
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")

# Opcional: resaltar zona si se desea (por ejemplo del 1000 al 2000)
plt.axvspan(1000, 2000, color='red', alpha=0.2, label='Zona anómala (ejemplo)')

# Activar cuadrícula
plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)

# Cambiar la posición de la leyenda
plt.legend(loc='upper left', bbox_to_anchor=(0.04, 0.1), borderpad=2, fancybox=True)

# Ajustar el layout
plt.tight_layout()

# Mostrar gráfico
plt.show()
