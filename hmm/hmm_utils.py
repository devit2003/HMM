import numpy as np
import librosa
import pickle
from scipy.stats import multivariate_normal

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    return mfcc.T

def load_hmm_model(path='hmm/model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def viterbi(obs, model):
    n_frames = obs.shape[0]
    n_states = model['n_states']
    mean, cov, trans = model['mean'], model['cov'], model['transition']
    log_probs = np.zeros((n_frames, n_states))
    backpointer = np.zeros((n_frames, n_states), dtype=int)
    for state in range(n_states):
        log_probs[0, state] = multivariate_normal.logpdf(obs[0], mean=mean, cov=cov)
    for t in range(1, n_frames):
        for state in range(n_states):
            probs = log_probs[t-1] + np.log(trans[:, state])
            backpointer[t, state] = np.argmax(probs)
            log_probs[t, state] = multivariate_normal.logpdf(obs[t], mean=mean, cov=cov) + np.max(probs)
    return np.sum(log_probs[-1])

def predict(models, feat_seq):
    scores = {label: viterbi(feat_seq, model) for label, model in models.items()}
    return max(scores, key=scores.get)
