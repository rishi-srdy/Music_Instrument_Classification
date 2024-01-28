import pickle
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import librosa

p_path = os.path.join('./pickles', 'conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
classes = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 'Double_bass', 'Flute', 'Hi_hat', 'Saxophone', 'Snare_drum', 'Violin_or_fiddle']

st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(config.model_path,
                                   custom_objects={'KerasLayer':hub.KerasLayer}
                                   )
  return model

def envelope(y,rate,threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    #rolling is used to check if the entire audio is dropped or not
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def save_wav_file(file, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())

model = load_model()
print(model.summary())

st.write("""# Musical Instrument Classification""")

mp3_file = st.file_uploader("Upload a wav file", type=["wav"])

# mp3_file = st_audiorec()

if mp3_file is not None:
   save_wav_file(mp3_file,'./database/')
   signal, rate = librosa.load('./database/'+mp3_file.name, sr=16000)
   mask = envelope(signal, rate, 0.0005)
   wavfile.write(filename='./cleaned_database'+mp3_file.name, rate=rate, data=signal[mask])
   rate, wav = wavfile.read('./cleaned_database'+mp3_file.name)
   st.audio(wav, format='audio/wav', start_time=0, sample_rate=rate)

   y_prob = []
   y_pred = []
   for i in range(0, wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        x = mfcc(sample, rate, numcep = config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
        x = (x-config.min)/(config.max - config.min)
        # st.text(x.shape[0])
        if config.mode == 'conv':
            x = x.reshape(1,x.shape[0], x.shape[1], 1)
        elif config.mode == 'time':
            x = np.expand_dims(x, axis=0)
        y_hat = model.predict(x)
        # print(y_hat)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))
        # y_pred.append(np.argmax(y_hat))
        # st.text(y_prob)
   y_pred = [classes[np.argmax(y)] for y in y_prob]
   st.text(y_pred[0])