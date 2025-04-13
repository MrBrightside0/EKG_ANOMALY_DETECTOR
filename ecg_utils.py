# ecg_utils.py

import numpy as np
import pandas as pd
import wfdb
import os
from config import FS, INPUT_LENGTH, ECG_PATH
import torch

def preprocess_single_signal_torch(signal, desired_length=5000):
    """
    Preprocesa una señal única para pasarla a un modelo PyTorch.
    - Recorta o rellena la señal.
    - Normaliza.
    - Devuelve un tensor [1, 1, desired_length].
    """
    # Recortar o rellenar
    if len(signal) < desired_length:
        signal = np.pad(signal, (0, desired_length - len(signal)), 'constant')
    else:
        signal = signal[:desired_length]

    # Normalización
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Convertir a tensor
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    return tensor

def load_metadata():
    df = pd.read_csv("C:\\Users\\cuent\\Desktop\\CNN HACKHATON\\ecg-system\\data\\ptbxl_database.csv")
    df['scp_codes'] = df['scp_codes'].apply(eval)
    return df

def load_ecg(record_name):
    path = os.path.join(ECG_PATH, record_name)
    record = wfdb.rdrecord(path)
    return record.p_signal

def extract_labels(df, top_labels):
    # Convertir diccionario a etiquetas binarias
    def encode(x):
        classes = list(x.keys())
        return [int(label in classes) for label in top_labels]

    df['labels'] = df['scp_codes'].apply(encode)
    return df

def get_top_labels(df, n=5):
    from collections import Counter
    all_codes = []
    for row in df['scp_codes']:
        all_codes.extend(row.keys())
    most_common = Counter(all_codes).most_common(n)
    return [code for code, _ in most_common]

def preprocess_signals(df, top_labels):
    X = []
    Y = []

    for _, row in df.iterrows():
        signal = load_ecg(row['filename_hr'])
        if signal.shape[0] < INPUT_LENGTH:
            continue  # saltar señales muy cortas

        sig = signal[:INPUT_LENGTH, 0]  # usar solo una derivación
        X.append(sig)
        Y.append(row['labels'])

    X = np.array(X)
    Y = np.array(Y)
    return X, Y
