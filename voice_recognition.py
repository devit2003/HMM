import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import joblib
from hmmlearn import hmm

def record_audio(duration=2, fs=16000, filename='command.wav'):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print("✅ Recording done.")

def extract_mfcc_from_file(filename):
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def load_models():
    import os
    model_dir = 'models'
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('.pkl'):
            label = fname.replace('_hmm.pkl', '')
            models[label] = joblib.load(os.path.join(model_dir, fname))
    return models

def predict_command(models, filename='command.wav'):
    mfcc = extract_mfcc_from_file(filename)
    scores = {label: model.score(mfcc) for label, model in models.items()}
    return max(scores, key=scores.get)
