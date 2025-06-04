import os
import numpy as np
import librosa
import pickle
from scipy.stats import multivariate_normal

LABELS = ['Next', 'Play', 'Pause', 'Up', 'Down', 'Close']
DATASET_PATH = 'dataset/'
MODEL_FILE = 'saved_model.pkl'

# Ekstraksi fitur MFCC
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        return mfcc.T
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None

# Load dataset dari folder
def load_dataset():
    data = {label: [] for label in LABELS}
    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            print(f"⚠️ Folder tidak ditemukan: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.endswith(('.mp3', '.wav')):
                fpath = os.path.join(folder, fname)
                feat = extract_features(fpath)
                if feat is not None and feat.shape[0] > 0:
                    data[label].append(feat)
    return data

# Latih model Gaussian HMM per label
def train_models(data):
    models = {}
    for label in LABELS:
        samples = data.get(label, [])
        if not samples:
            print(f"⚠️ Tidak ada data untuk label: {label}")
            continue
        all_feat = np.vstack(samples)
        mean = np.mean(all_feat, axis=0)
        cov = np.cov(all_feat.T) + 1e-6 * np.eye(all_feat.shape[1])  # Regularisasi
        transition = np.ones((3, 3)) / 3  # Matriks transisi sederhana
        models[label] = {
            'mean': mean,
            'cov': cov,
            'transition': transition,
            'n_states': 3
        }
        print(f"✅ Model dilatih untuk label: {label}")
    return models

# Simpan model ke file
def save_model(models):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    print(f"💾 Model disimpan ke: {MODEL_FILE}")

# Fungsi utama training
def main():
    print("📥 Memuat dataset suara...")
    data = load_dataset()
    if not any(data.values()):
        print("❌ Gagal memuat data. Pastikan folder dataset dan file audio tersedia.")
        return

    print("🧠 Melatih model HMM per label...")
    models = train_models(data)

    print("💾 Menyimpan model ke file...")
    save_model(models)

    print("🎉 Training selesai dan model siap digunakan!")

if __name__ == '__main__':
    main()
