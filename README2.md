# 🫀 Heart Murmur Detection Using Audio Signals & Deep Learning

An end-to-end machine learning project that detects **heart murmurs from heart sound audio recordings** using **audio preprocessing, MFCC feature extraction, and an LSTM deep learning model**, deployed as an interactive **Streamlit web application**.

---

## 🚀 Project Overview

Heart murmurs are abnormal sounds during a heartbeat that may indicate underlying heart conditions. Early detection is crucial for timely diagnosis and treatment.  
This project uses **audio signal processing and deep learning** to automatically classify heart sounds and assist in murmur detection.

The system covers the **complete ML lifecycle**:
- Data loading & preprocessing
- Feature extraction
- Model training with imbalance handling
- Model deployment using Streamlit
- Cloud-based model hosting via Hugging Face

---

## 🎯 Aim

To build and deploy a deep learning-based system that can analyze heart sound recordings and accurately detect heart murmurs from audio signals.

---

## 📂 Dataset

- **Name:** Heartbeat Sound Dataset  
- **Source:** Kaggle (Open Source)  
- **Link:** https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound  
- **Format:** `.wav` audio files  

### Classes Used
- Artifact  
- Murmur  
- Normal  
- Extrahls  
- Extrastole  

> Unlabelled audio files are used only for testing and inference.

---

## 🛠️ Tools & Technologies

- **Programming Language:** Python  
- **Audio Processing:** Librosa, NumPy  
- **Deep Learning:** TensorFlow, Keras  
- **Model Architecture:** LSTM  
- **Feature Extraction:** MFCC  
- **Web Framework:** Streamlit  
- **Model Hosting:** Hugging Face Hub  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub  

---

## 🧠 System Architecture

```
Audio Input (.wav / .mp3)
        ↓
Audio Preprocessing
(Sampling Rate + Duration Fix)
        ↓
MFCC Feature Extraction
        ↓
LSTM Deep Learning Model
        ↓
Prediction Output
        ↓
Streamlit Web Interface
```

---

## 🔄 Project Workflow

1. Load heart sound audio files  
2. Standardize sampling rate (22,050 Hz)  
3. Fix audio duration (10 seconds)  
4. Extract MFCC features  
5. Handle class imbalance using class weights  
6. Train LSTM model  
7. Evaluate model performance  
8. Upload trained model to Hugging Face Hub  
9. Deploy inference pipeline using Streamlit  

---

## 📊 Handling Class Imbalance

The dataset is naturally imbalanced.  
Instead of oversampling or undersampling, **class weights** are used during training so that minority classes contribute more to the loss function.

This approach:
- Preserves original data  
- Reduces overfitting  
- Is suitable for medical datasets  

---

## 🌐 Web Application (Streamlit)

The Streamlit app allows users to:
- Upload heart sound audio files (`.wav` / `.mp3`)  
- Visualize the audio waveform  
- Run real-time murmur prediction  
- View model confidence scores  

The trained model is **loaded dynamically from Hugging Face Hub** for scalability and version control.

---

## 📁 Project Structure

```
├── app.py
├── model/
│   └── loader.py
├── audio/
│   └── preprocessing.py
├── ui/
│   └── visualizations.py
├── utils/
│   └── logger.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 📌 Key Learning Outcomes

- Working with audio data for ML  
- Audio preprocessing techniques  
- MFCC feature extraction  
- Sequence modeling using LSTM  
- Handling class imbalance  
- Model deployment with Streamlit  
- Cloud-based model hosting with Hugging Face  

---

## 🔮 Future Improvements

- Add CNN-based spectrogram models  
- Improve evaluation using precision & recall  
- Add model explainability  
- Support real-time audio recording  
- Provide REST API for predictions  

---

## 👨‍🎓 Ideal For

- Machine Learning Students  
- Deep Learning Beginners  
- Medical AI Projects  
- Audio Signal Processing Learning  

---

## 📜 License

This project is open-source and intended for **educational purposes only**.

---

## ⭐ Acknowledgements

- Kaggle for the open-source dataset  
- Librosa & TensorFlow communities  
- Streamlit & Hugging Face teams  

---

> If you found this project useful, feel free to ⭐ the repository!
