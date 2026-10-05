import numpy as np
import streamlit as st
import torch

from model.model_loader import load_model
from ui.visualization import plot_waveform
from utils.logger import set_logger
from audio.preprocessing import load_audio, extract_mfcc

logger = set_logger("StreamlitAPP")

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
model = load_model()
model.to(device)


st.title("🎵 Heart Murmur Detection with LSTM")

uploaded_file = st.file_uploader(
    "Upload a Heart Sound (WAV/MP3)", type = ["wav", "mp3"]
)

if uploaded_file is not None:
    try:
        y, sr = load_audio(uploaded_file)
        
        st.subheader("Waveform of Input Sound")
        fig = plot_waveform(y, sr)
        st.pyplot(fig)
        
        x_input = extract_mfcc(y, sr)

        x_input = torch.tensor(
            x_input, dtype = torch.float32
        )
        x_input = x_input.transpose(1, 2)

        x_input = x_input.to(device)

        with torch.no_grad():
            prediction = model(x_input)

        prediction = prediction.cpu()

        # Convert logits to probabilities
        probabilities = torch.softmax(prediction, dim=1)

        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Probability of the predicted class
        predicted_probability = probabilities[0, predicted_class].item()
        st.subheader("🔮 Prediction Result")
        if predicted_class == 0:
            st.write("Artifact")
        elif predicted_class == 1:
            st.write("Murmur") 
        else:
            st.write("Normal")
        
        st.write(f"Prediction Probability: {predicted_probability:.2%}")
    except Exception as e:
        logger.exception("Inference pipeline failed") 
        st.error("⚠️ An error occurred while processing the audio file.")   
                      