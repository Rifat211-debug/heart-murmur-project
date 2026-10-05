import librosa
import numpy as np
from utils.logger import set_logger
from config import N_MFCC, SAMPLE_RATE

logger = set_logger("AudioPreprocessing")

def load_audio(uploaded_file):
    try:
        logger.info("Loading Audio File")
        y, sr = librosa.load(uploaded_file, sr = SAMPLE_RATE)
        return y, sr  
    except Exception as e:
        logger.exception("Audio Loading Failed")
        raise RuntimeError("Invalid or corrupted audion file") from e

def extract_mfcc(y, sr):
    try:
        logger.info("Extracting MFCC Features")
        mfcc = librosa.feature.mfcc(
            y = y, sr = sr, n_mfcc = N_MFCC
        )   
        mfcc_scaled = np.mean(mfcc.T, axis = 0)
        x_input = np.expand_dims(mfcc_scaled, axis = 0)
        x_input = np.expand_dims(x_input, axis = 2)
        return x_input
    except Exception as e:
        logger.info("MFCC extraction failed")
        raise RuntimeError("Feature Extraction Failed") from e
      