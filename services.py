from moviepy import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import os


import torch
import torch.nn as nn
import joblib
import librosa
import numpy as np
# =========================
# Internal Memory Storage
# =========================
_latest_video_result = None
_latest_project_result = None



MAX_PAD_LEN = 174
SR = 22050
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128 * 16 * 21, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

model = DeeperCNN(NUM_CLASSES).to(DEVICE)


try:
    model.load_state_dict(torch.load('ser_cnn_model.pth', map_location=DEVICE))
except:
    model = torch.load('ser_cnn_model.pth', map_location=DEVICE)

model.eval()
le = joblib.load('label_encoder.pkl')


def predict_emotion(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=SR)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        mel_spectrogram = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        current_len = log_mel_spectrogram.shape[1]
        if current_len > MAX_PAD_LEN:
            feature = log_mel_spectrogram[:, :MAX_PAD_LEN]
        elif current_len < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - current_len
            feature = np.pad(log_mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            feature = log_mel_spectrogram

        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(feature_tensor)
            predicted_index = torch.argmax(output, dim=1).cpu().item()

        return le.inverse_transform([predicted_index])[0]
    except Exception as e:
        return f"Error: {e}"

# =========================
# Save / Get Video Result
# =========================
def save_video_result(result):
    global _latest_video_result
    _latest_video_result = result


def get_video_result():
    return _latest_video_result


# =========================
# Save / Get Project Result
# =========================
def save_project_result(result):
    global _latest_project_result
    _latest_project_result = result


def get_project_result():
    return _latest_project_result


# =========================
# Video → Audio + Speech Recognition
# =========================
def process_video(video_path):
    global _latest_video_result
    try:
        # 1️⃣ Extract Audio
        video = VideoFileClip(video_path)
        temp_mp3 = "temp.mp3"
        video.audio.write_audiofile(temp_mp3, codec="mp3", logger=None)
        video.close()

        # 2️⃣ Convert to WAV
        sound = AudioSegment.from_mp3(temp_mp3)
        sound = sound.set_channels(1).set_frame_rate(22050)

        audio_path = "audio.wav"
        sound.export(audio_path, format="wav")

        # 3️⃣ Speech Recognition
        recognizer = sr.Recognizer()
        full_text = ""

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        emotion_result = predict_emotion(audio_path)
        _latest_video_result = {
            "text": text,
            "emotion": emotion_result,
            "audio_path": audio_path
        }
        return _latest_video_result
    except Exception as e:
        print(f"General Error: {e}")
        return None

def get_video_result():
    return _latest_video_result

