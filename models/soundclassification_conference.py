# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import argparse
from pydub import AudioSegment
import soundfile as sf
import os, sys, glob
import pickle

AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"
AudioSegment.ffprobe = os.getcwd()+ "\\ffprobe.exe"

def wav_transform (input_file_path):
  wavfile = input_file_path.replace ('m4a', 'wav')
  sound = AudioSegment.from_file(input_file_path, format='m4a')
  wav_file = sound.export (wavfile, format='wav')
  return wav_file

def preprocess(input_file_path):
  audio, sr = librosa.load(path = input_file_path, sr=22050)
  audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
  zcr=librosa.feature.zero_crossing_rate(audio)
  zcritem=zcr.flatten()
  zcritem=zcritem.reshape(1,44)
  cent = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
  centitem=cent.flatten()
  centitem=cent.reshape(1,44)
  mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=231,norm=np.inf)
  mels_db = librosa.power_to_db(S=mels, ref=1.0)
  mels_out=mels_db.reshape((1,128, 16, 6))
  cens = librosa.feature.chroma_cens (y = audio, sr = sr)
  cens = cens.reshape (1, 12*44)
  return [mels_out,cens,centitem, zcritem]

def classify(input_file_path):
  #pmeclasses=["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
  pmeclasses=["Electric Percussive Drill", "Handheld percussive breakers <10kg","Handheld percussive breakers >10kg","Others"]
  testfeat = preprocess(input_file_path)
  loaded_model = tf.keras.models.load_model('soundmodel_conference.h5') 
  extra_model = pickle.load(open('below and over 10 conference.pkl', 'rb'))
  preds=loaded_model.predict(testfeat)
  indmax = np.argmax(preds[0])
  if pmeclasses[indmax] in ["Handheld percussive breakers >10kg", "Handheld percussive breakers <10kg"]:
    preds = extra_model.predict (testfeat [0].reshape(1, 128*16*6))
    return preds
  else: return pmeclasses[indmax]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='file_location')
    args = parser.parse_args()
    wav_file = wav_transform (args.location)
    predic = classify(wav_file)
    print (predic)

if __name__ == "__main__":
    main ()
