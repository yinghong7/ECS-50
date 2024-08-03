import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import argparse
from pydub import AudioSegment
import soundfile as sf
import os, sys, glob
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
  return [mels_out,centitem,zcritem], np.array(cens.flatten())

def classify(input_file_path):
  #pmeclasses=["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
  pmeclasses=['Electrical drill', 'Handheld less 10', 'Others', 'Percussive over 10']
  #pmeclasses=["Electric Percussive Drill", "Handheld percussive breakers <10kg","Handheld percussive breakers >10kg","Others"]
  testfeat, cens_feat = preprocess(input_file_path)
  loaded_model = tf.keras.models.load_model('soundmodel.h5') #path for AI model
  #loaded_model = tf.keras.models.load_model('test_v3.h5') #conference room setting
  extra_model = pickle.load(open('below and over 10.pkl', 'rb'))
  preds=loaded_model.predict(testfeat)
  indmax = np.argmax(preds[0])
  #return preds, indmax, pmeclasses[indmax]
  if pmeclasses[indmax] == "Handheld percussive breakers >10kg":
    preds = extra_model.predict (cens_feat.reshape(1, -1))
    return preds[0]
  else: return pmeclasses[indmax]

#folderpath = "C:\\Users\\Ying.Hong\\OneDrive - Arup\\Breaker_audio_clip\\91_Ecolab_10th May_splited"
folderpath = "C:\\Users\\Ying.Hong\\OneDrive - Arup\\Breaker_audio_clip\\92_New acute hospital"
y_predicted, y_true = [], []
for wav_file in glob.glob(os.path.join(folderpath, '*')):
   location = wav_file.split('\\')[-1].split('_')[1]
   if location == 'floor removal':
    y_pred = classify(wav_file)
    y_tr = wav_file.split('\\')[-1].split('_')[0]
    y_predicted.append (y_pred)
    y_true.append (y_tr)


# display
cnf_matrix=confusion_matrix(y_true, y_predicted, labels = ['Electrical drill', 'Handheld less 10', 'Others', 'Percussive over 10'])
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# false positive rate
FPR = FP/(FP+TN)
print('FPR: '+str(FPR*100))
# true positive rate
TPR = TP/(TP+FN)
print('TPR: '+str(TPR*100))

disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=['Electrical drill', 'Handheld less 10', 'Others', 'Percussive over 10'])
disp.plot()
plt.show()