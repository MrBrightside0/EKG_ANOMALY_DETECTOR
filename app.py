import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from ecg_utils import preprocess_single_signal_torch
from config import INPUT_LENGTH, NUM_CLASSES
from predict import ECG_CNN, get_top_labels, load_metadata
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# App Config
st.set_page_config(page_title="ECG Diagnóstico", layout="centered")
st.title("🫀 Sistema de diagnóstico de ECG")

# Model Charging
model = ECG_CNN()
model.load_state_dict(torch.load(r"C:\Users\cuent\Desktop\CNN HACKHATON\ecg-system\models\ecg_cnn_model.pt", map_location=torch.device("cpu")))
model.eval()

# Getting labels
df = load_metadata()
top_labels = get_top_labels(df, NUM_CLASSES)

# Upload Files
uploaded_dat = st.file_uploader("📁 Sube archivo .dat", type="dat")
uploaded_hea = st.file_uploader("📁 Sube archivo .hea", type="hea")

# Patien's Data forms
st.sidebar.header("📝 Datos del paciente")
cama_num = st.sidebar.text_input("Número de cama")
expediente_num = st.sidebar.text_input("Número de expediente")
nombre_completo = st.sidebar.text_input("Nombre completo")
hora = st.sidebar.text_input("Hora")
antecedentes = st.sidebar.text_area("Antecedentes")

def generate_pdf(data, image):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Agdd Text
    c.drawString(100, 750, "📝 Datos del paciente:")
    y_position = 730
    for line in data.split("\n"):
        c.drawString(100, y_position, line)
        y_position -= 20
    
    c.showPage()
    
    # Insertar imagen
    c.drawImage(image, 100, 450, width=400, height=200)
    
    c.save()
    buffer.seek(0)
    return buffer

if uploaded_dat and uploaded_hea:
    # Obtener nombre original desde .hea
    hea_content = uploaded_hea.read().decode("utf-8")
    first_line = hea_content.splitlines()[0]
    record_name = first_line.split()[0]

    # Guardar archivos con el nombre correcto
    with open(f"{record_name}.hea", "w") as f:
        f.write(hea_content)

    with open(f"{record_name}.dat", "wb") as f:
        f.write(uploaded_dat.read())

    try:
        record = wfdb.rdrecord(record_name)
        signal = record.p_signal[:, 0]

        # Gráfico de la señal ECG
        st.subheader("📊 Señal ECG")
        
        # Crear la gráfica
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal[:INPUT_LENGTH])
        
        # Etiquetas de los ejes
        ax.set_xlabel("Tiempo (s)")  # Cambiar a "S" en el eje X
        ax.set_ylabel("MV (Milivoltios)")  # Cambiar a "MV" en el eje Y
        ax.set_title("ECG - Derivación 1")

        # Agregar la cuadrícula
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        # Guardar la imagen para el PDF
        image_path = "ecg_image.png"
        fig.savefig(image_path)

        st.pyplot(fig)

        if len(signal) < INPUT_LENGTH:
            st.warning("La señal es muy corta para evaluar.")
        else:
            x = torch.tensor(signal[:INPUT_LENGTH], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(x)
                output = output.squeeze().numpy()

            st.subheader("🧠 Diagnóstico")
            diagnostico = ""
            anormal = False
            for label, prob in zip(top_labels, output):
                if prob > 0.5:
                    st.markdown(f"✅ **{label}** detectado con {prob:.2f} de confianza")
                    diagnostico += f"{label}: {prob:.2f} ✅\n"
                    anormal = True
                else:
                    st.markdown(f"❌ {label}: {prob:.2f}")
                    diagnostico += f"{label}: {prob:.2f} ❌\n"


            if anormal:
                st.success("🚨 ¡Se detectaron posibles anomalías!")
            else:
                st.success("✅ No se detectaron patologías significativas.")

            # Zona resaltada (opcional)
            st.subheader("📍 Zona resaltada")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(signal[:INPUT_LENGTH])
            ax.axvspan(200, 400, color='red', alpha=0.3, label="Zona sospechosa")
            ax.set_title("ECG con zona resaltada")

            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("MV Minivolts")
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
            ax.legend()

            # Guardar la imagen resaltada
            highlighted_image_path = "ecg_highlighted_image.png"
            fig.savefig(highlighted_image_path)

            st.pyplot(fig)

            # Mostrar datos del paciente
            if cama_num and expediente_num and nombre_completo and hora and antecedentes:
                st.subheader("📋 Datos del paciente")
                st.write(f"Número de cama: {cama_num}")
                st.write(f"Número de expediente: {expediente_num}")
                st.write(f"Nombre completo: {nombre_completo}")
                st.write(f"Hora: {hora}")
                st.write(f"Antecedentes: {antecedentes}")

                # Generar contenido del archivo PDF
                patient_data = f"\n"
                patient_data += f"Número de cama: {cama_num}\n"
                patient_data += f"Número de expediente: {expediente_num}\n"
                patient_data += f"Nombre completo: {nombre_completo}\n"
                patient_data += f"Hora: {hora}\n"
                patient_data += f"Antecedentes: {antecedentes}\n\n"
                patient_data += f"🧠 Diagnóstico:\n\n"
                patient_data += diagnostico

                # Generar el PDF
                pdf_buffer = generate_pdf(patient_data, highlighted_image_path)

                # Descargar el archivo PDF
                st.download_button(
                    label="📥 Descargar diagnóstico en PDF",
                    data=pdf_buffer,
                    file_name="diagnostico_ecg.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
