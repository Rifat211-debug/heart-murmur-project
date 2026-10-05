import numpy as np
import streamlit as st

from model.model_loader import load_model
from ui.visualization import plot_waveform
from utils.logger import set_logger
from audio.preprocessing import load_audio, extract_mfcc

logger = set_logger("StrealitAPP")

model = load_model()

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
        
        prediction = model.predict(x_input)
        predicted_class = np.argmax(prediction, axis = 1)[0]
            #{0: 'artifact', 1: 'murmur', 2: 'normal'
        st.subheader("🔮 Prediction Result")
        if predicted_class == 0:
            st.write("Artifact")
        elif predicted_class == 1:
            st.write("Murmur") 
        else:
            st.write("Normal")
        
        st.write(f"Prediction Score : {prediction}") 
    except Exception as e:
        logger.info("Inference pipeline failed") 
        st.error("⚠️ An error occurred while processing the audio file.")   
                      