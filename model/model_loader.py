import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from huggingface_hub import hf_hub_download

from config import HF_REPO_ID, HF_MODEL_FILENAME
from utils.logger import set_logger


logger = set_logger("ModelLoader")


class CNNLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, 2048, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool1d(2, 2)
        self.bn1 = nn.BatchNorm1d(2048)

        self.conv2 = nn.Conv1d(
            2048, 1024, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool1d(2, 2)
        self.bn2 = nn.BatchNorm1d(1024)

        self.conv3 = nn.Conv1d(
            1024, 512, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool1d(2, 2)
        self.bn3 = nn.BatchNorm1d(512)

        self.lstm1 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True
        )

        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Last sequence output
        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


@st.cache_resource
def load_model():
    try:
        logger.info("Downloading model from Hugging Face Hub")
        model_path = hf_hub_download(
            repo_id = HF_REPO_ID,
            filename = HF_MODEL_FILENAME,
            repo_type = "model"
        )
        logger.info("Loading PyTorch model")

        model = CNNLSTM()

        model.load_state_dict(
            torch.load(
                model_path,
                map_location = torch.device("cpu")
            )
        )

        model.eval()

        logger.info("Model loaded successfully")

        return model

    except Exception as e:
        logger.exception("Failed to load model")
        st.error("❌ Failed to load the ML model. Please try again later.")
        raise e