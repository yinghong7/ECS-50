import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, det_curve, ConfusionMatrixDisplay
import pickle
import seaborn as sns


# load background noise features
def load_pme_site_a ():
    features = pickle.load(open(os.getcwd() + '\\PME_audio_features\\mel_list_a.pkl', 'rb')).tolist()
    zcr_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\zcr_list_a.pkl', 'rb')).tolist()
    cent_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\cent_list_a.pkl', 'rb')).tolist()
    label = pickle.load(open(os.getcwd() + '\\PME_audio_features\\label_list_a.pkl', 'rb'))
    cens_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\cens_list_a.pkl', 'rb')).tolist()
    length = len(label)
    features = np.array(features).reshape (length,128,16,6)
    zcr_list = np.array(zcr_list).reshape (length,44)
    cent_list = np.array(cent_list).reshape (length,44)
    cens_list = np.array(cens_list).reshape (length, 12*44)
    return features, zcr_list, cent_list, cens_list, label

def load_pme_site_b ():
    features = pickle.load(open(os.getcwd() + '\\PME_audio_features\\mel_list_b.pkl', 'rb')).tolist()
    zcr_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\zcr_list_b.pkl', 'rb')).tolist()
    cent_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\cent_list_b.pkl', 'rb')).tolist()
    label = pickle.load(open(os.getcwd() + '\\PME_audio_features\\label_list_b.pkl', 'rb'))
    cens_list = pickle.load(open(os.getcwd() + '\\PME_audio_features\\cens_list_b.pkl', 'rb')).tolist()
    length = len(label)
    features = np.array(features).reshape (length,128,16,6)
    zcr_list = np.array(zcr_list).reshape (length,44)
    cent_list = np.array(cent_list).reshape (length,44)
    cens_list = np.array(cens_list).reshape (length, 12*44)
    return features, zcr_list, cent_list, cens_list, label

# load pme audio features (onsite)
def load_back ():
    features = pickle.load(open(os.getcwd() + '\\feature\\mel_list_esc.pkl', 'rb')).tolist()
    zcr_list = pickle.load(open(os.getcwd() + '\\feature\\zcr_list_esc.pkl', 'rb')).tolist()
    cent_list = pickle.load(open(os.getcwd() + '\\feature\\cent_list_esc.pkl', 'rb')).tolist()
    cens_list = pickle.load(open(os.getcwd() + '\\feature\\cens_list_esc.pkl', 'rb')).tolist()
    length = len(zcr_list)
    label = ['Others'] * length
    features = np.array(features).reshape (length,128,16,6)
    zcr_list = np.array(zcr_list).reshape (length,44)
    cent_list = np.array(cent_list).reshape (length,44)
    cens_list = np.array(cens_list).reshape (length, 12*44)
    return features, zcr_list, cent_list, cens_list, label

def ResNet34_v2 (f1, f2, f3, f4, classes = 4):
    # Step 1 (Setup Input Layer)
    x_input = keras.layers.Input(shape = f1[0].shape)
    y_input = keras.layers.Input(shape = f2[0].shape)
    z_input = keras.layers.Input(shape = f3[0].shape)
    l_input = keras.layers.Input(shape = f4[0].shape)
    cnn=keras.layers.Conv2D(16,3, activation='relu',padding='same', kernel_regularizer='l2')(x_input)
    cnn = keras.layers.MaxPooling2D()(cnn)
    cnn=keras.layers.Conv2D(32,3, activation='relu',padding='same', kernel_regularizer='l2')(cnn)
    cnn = keras.layers.MaxPooling2D()(cnn)
    cnn=keras.layers.Flatten()(cnn)
    num = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num")(y_input)
    num = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num")(num)
    num2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num2")(z_input)
    num2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num2")(num2)
    num3 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num3")(l_input)
    num3 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num3")(num3)
    full = tf.keras.layers.concatenate([cnn,num,num2], name="concat_full")
    full= keras.layers.Dense(32, activation='relu', kernel_regularizer='l2')(full)
    full = keras.layers.Dense(classes, activation='softmax', kernel_regularizer='l2')(full)
    model = keras.models.Model(inputs = [x_input, y_input, z_input, l_input], outputs = full, name = "ResNet34_v2")
    return model

# load data
feature_pme_a, zcr_pme_a, cent_pme_a, cens_pme_a, label_pme_a = load_pme_site_a ()
feature_pme_b, zcr_pme_b, cent_pme_b, cens_pme_b, label_pme_b = load_pme_site_b ()
feature_back, zcr_back, cent_back, cens_back, label_back = load_back ()

features = np.vstack ((feature_pme_a, feature_pme_b, feature_back))
zcr_list = np.vstack ((zcr_pme_a, zcr_pme_b, zcr_back))
cent_list = np.vstack ((cent_pme_a, cent_pme_b, cent_back))
cens_list = np.vstack ((cens_pme_a, cens_pme_b, cens_back))

label = label_pme_a + label_pme_b + label_back
lb = LabelBinarizer()
labelz = lb.fit_transform (label)
print (np.unique (label))

# ResNet model
test_model = ResNet34_v2(features, cent_list, zcr_list)
test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
TEST_SIZE = 0.3
BATCH_SIZE = 1
EPOCHS = 50
X_train, X_test,X2_train,X2_test,X3_train,X3_test,y_train, y_test = train_test_split(features,cent_list,zcr_list,labelz, test_size=TEST_SIZE)
history = test_model.fit(x=[X_train,X2_train,X3_train], y=y_train, validation_data=([X_test, X2_test,X3_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
test_model.save ('test_s_01.h5')


pmeclasses= ['Electrical drill' 'Handheld less 10' 'Others' 'Percussive over 10']

# SVM specific
def find(ls, pme_tool):
    return [i for i, x in enumerate(ls) if x == pme_tool]

over10 = find(label, 'Percussive over 10')
under10 = find(label, 'Handheld less 10')

