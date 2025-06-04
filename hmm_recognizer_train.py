import os
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import multivariate_normal

LABELS = ['Next', 'Play', 'Pause', 'Up', 'Down', 'Close']
DATASET_PATH = 'dataset/'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'hmm_model.pkl')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        return mfcc.T
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_dataset():
    data = {label: [] for label in LABELS}
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} does not exist")
        return data
    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            print(f"Folder for {label} does not exist")
            continue
        for fname in os.listdir(folder):
            if fname.endswith(('.mp3', '.wav')):
                fpath = os.path.join(folder, fname)
                feat = extract_features(fpath)
                if feat is not None and feat.shape[0] > 0:
                    data[label].append(feat)
    return data

def estimate_gaussian_models(data):
    models = {}
    for label in LABELS:
        if not data[label]:
            print(f"No data for label {label}")
            continue
        all_feat = np.vstack(data[label])
        mean = np.mean(all_feat, axis=0)
        cov = np.cov(all_feat.T) + 1e-6 * np.eye(all_feat.shape[1])
        n_states = 3
        transition = np.ones((n_states, n_states)) / n_states
        models[label] = {
            'mean': mean,
            'cov': cov,
            'transition': transition,
            'n_states': n_states
        }
    return models

def viterbi(obs, model):
    n_frames = obs.shape[0]
    n_states = model['n_states']
    mean, cov, trans = model['mean'], model['cov'], model['transition']
    log_probs = np.zeros((n_frames, n_states))
    backpointer = np.zeros((n_frames, n_states), dtype=int)
    for state in range(n_states):
        log_probs[0, state] = multivariate_normal.logpdf(obs[0], mean=mean, cov=cov, allow_singular=True)
    for t in range(1, n_frames):
        for state in range(n_states):
            probs = log_probs[t-1] + np.log(trans[:, state])
            backpointer[t, state] = np.argmax(probs)
            log_probs[t, state] = multivariate_normal.logpdf(obs[t], mean=mean, cov=cov, allow_singular=True) + np.max(probs)
    path = np.zeros(n_frames, dtype=int)
    path[-1] = np.argmax(log_probs[-1])
    for t in range(n_frames-2, -1, -1):
        path[t] = backpointer[t+1, path[t+1]]
    return np.sum(log_probs[-1])

def predict(models, feat_seq):
    if feat_seq is None or feat_seq.shape[0] == 0:
        return None
    scores = {}
    for label in models:
        score = viterbi(feat_seq, models[label])
        scores[label] = score
    return max(scores, key=scores.get)

def evaluate(models, test_data):
    y_true, y_pred = [], []
    for label in LABELS:
        for feat in test_data.get(label, []):
            pred = predict(models, feat)
            if pred:
                y_true.append(label)
                y_pred.append(pred)
    if not y_true:
        print("No valid test data available")
        return
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("🔁 Memuat data dan melatih model HMM...")
all_data = load_dataset()

# Split: 20% testing, 80% training
train_data, test_data = {}, {}
for label in LABELS:
    samples = all_data.get(label, [])
    if len(samples) < 2:
        print(f"Insufficient samples for {label}")
        continue
    split_idx = max(1, int(len(samples) * 0.2))
    test_data[label] = samples[:split_idx]
    train_data[label] = samples[split_idx:]

if not any(train_data.values()):
    print("No training data available")
    exit(1)

models = estimate_gaussian_models(train_data)
if not models:
    print("No models trained")
    exit(1)

print("\n📊 Evaluasi model...")
evaluate(models, test_data)

# Simpan model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(models, f)
print(f"💾 Model HMM disimpan di {MODEL_PATH}")