import os
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import multivariate_normal

LABELS = ['Next', 'Play', 'Pause', 'Up', 'Down', 'Close']
DATASET_PATH = 'dataset/'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'hmm_model.pkl')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2]).T
        return combined
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_dataset():
    data = {label: [] for label in LABELS}
    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if fname.endswith(('.mp3', '.wav')):
                feat = extract_features(os.path.join(folder, fname))
                if feat is not None and feat.shape[0] > 0:
                    data[label].append(feat)
    return data

def estimate_gaussian_models(data, n_states=4):
    models = {}
    for label in LABELS:
        if not data[label]: continue
        all_feat = np.vstack(data[label])
        means, covs = [], []
        kmeans_idx = np.random.choice(len(all_feat), n_states, replace=False)
        for idx in kmeans_idx:
            sub = all_feat[np.random.choice(len(all_feat), 100, replace=True)]
            means.append(np.mean(sub, axis=0))
            covs.append(np.cov(sub.T) + 1e-6 * np.eye(sub.shape[1]))
        transition = np.ones((n_states, n_states)) / n_states
        models[label] = {
            'means': means,
            'covs': covs,
            'transition': transition,
            'n_states': n_states
        }
    return models

def viterbi(obs, model):
    n_frames, n_states = obs.shape[0], model['n_states']
    means, covs, trans = model['means'], model['covs'], model['transition']
    log_probs = np.full((n_frames, n_states), -np.inf)
    backpointer = np.zeros((n_frames, n_states), dtype=int)

    for s in range(n_states):
        log_probs[0, s] = multivariate_normal.logpdf(obs[0], mean=means[s], cov=covs[s], allow_singular=True)

    for t in range(1, n_frames):
        for s in range(n_states):
            probs = log_probs[t-1] + np.log(trans[:, s])
            log_probs[t, s] = np.max(probs) + multivariate_normal.logpdf(obs[t], mean=means[s], cov=covs[s], allow_singular=True)
            backpointer[t, s] = np.argmax(probs)

    return np.max(log_probs[-1])

def predict(models, feat_seq):
    scores = {label: viterbi(feat_seq, model) for label, model in models.items()}
    return max(scores, key=scores.get)

def evaluate(models, test_data):
    y_true, y_pred = [], []
    for label in LABELS:
        for feat in test_data[label]:
            pred = predict(models, feat)
            y_true.append(label)
            y_pred.append(pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Training logic
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
print("🔁 Loading dataset...")
all_data = load_dataset()

train_data, test_data = {}, {}
for label in LABELS:
    train_data[label], test_data[label] = train_test_split(all_data[label], test_size=0.2, random_state=42)

print("🧠 Estimating HMM models...")
models = estimate_gaussian_models(train_data)

print("📊 Evaluating...")
evaluate(models, test_data)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(models, f)
print(f"💾 Model saved to {MODEL_PATH}")
