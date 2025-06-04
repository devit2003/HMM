from flask import Flask, render_template, request, jsonify
import os
from hmm.hmm_utils import extract_features, load_hmm_model, predict
import sounddevice as sd
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MUSIC_FOLDER = 'static/music'
HMM_MODEL = load_hmm_model()

@app.route('/')
def index():
    songs = [f for f in os.listdir(MUSIC_FOLDER) if f.endswith('.mp3')]
    return render_template('index.html', songs=songs)

@app.route('/record', methods=['POST'])
def record():
    duration = 2  # seconds
    fs = 16000
    filename = os.path.join(UPLOAD_FOLDER, 'command.wav')
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)

    feat = extract_features(filename)
    command = predict(HMM_MODEL, feat)
    return jsonify({'command': command})

if __name__ == '__main__':
    app.run(debug=True)
