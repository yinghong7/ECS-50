import numpy as np
import json
import os, glob
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, det_curve, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def open_json (folder, equipment):
    folderpath = os.getcwd() + folder
    chroma = json.load (open (folderpath + '\\' + equipment + '_chroma.json'))
    energy = json.load (open (folderpath + '\\' + equipment + '_energy.json'))
    mfcc = json.load (open (folderpath + '\\' + equipment + '_mfcc.json'))
    spectral = json.load (open (folderpath + '\\' + equipment + '_spectralSpread.json'))
    zcr = json.load (open (folderpath + '\\' + equipment + '_zcr.json'))
    return np.array(chroma), np.array(energy), np.array(mfcc), np.array(spectral), np.array(zcr)

def feature_combine (folder, equipment_lst, type):
    chroma_f, energy_f, mfcc_f, spectral_f, zcr_f = np.zeros ((1, 44, 12)), np.zeros ((1, 44)), np.zeros ((1, 44, 13)), np.zeros ((1, 44)), np.zeros ((1, 44))
    for i in equipment_lst:
        chroma, energy, mfcc, spectral, zcr = open_json(folder, i)
        chroma_f = np.append (chroma_f, chroma, axis = 0)
        energy_f = np.append (energy_f, energy, axis = 0)
        mfcc_f = np.append (mfcc_f, mfcc, axis = 0)
        spectral_f = np.append (spectral_f, spectral, axis = 0)
        zcr_f = np.append (zcr_f, zcr, axis = 0)
    chroma_f = np.delete(chroma_f, (0), axis = 0)
    energy_f = np.delete(energy_f, (0), axis = 0)
    mfcc_f = np.delete(mfcc_f, (0), axis = 0)
    spectral_f = np.delete(spectral_f, (0), axis = 0)
    zcr_f = np.delete(zcr_f, (0), axis = 0)
    return chroma_f, energy_f, mfcc_f, spectral_f, zcr_f

def ResNet34_v2 (mfcc, chroma, energy, spectral, zcr, classes = 4):
    # Step 1 (Setup Input Layer)
    mfcc_input = keras.layers.Input(shape = mfcc[0].shape)
    chroma_input = keras.layers.Input(shape = chroma[0].shape)
    energy_input = keras.layers.Input(shape = energy[0].shape)
    spectral_input = keras.layers.Input(shape = spectral[0].shape)
    zcr_input = keras.layers.Input(shape = zcr[0].shape)
    # conv1d layers for 3d inputs
    cnn1 = keras.layers.Conv1D(16,3, activation='relu',padding='same', kernel_regularizer='l2')(mfcc_input)
    cnn1 = keras.layers.MaxPooling1D()(cnn1)
    cnn1 = keras.layers.Conv1D(32,3, activation='relu',padding='same', kernel_regularizer='l2')(cnn1)
    cnn1 = keras.layers.MaxPooling1D()(cnn1)
    cnn1 = keras.layers.Flatten()(cnn1)
    cnn2 = keras.layers.Conv1D(16,3, activation='relu',padding='same', kernel_regularizer='l2')(chroma_input)
    cnn2 = keras.layers.MaxPooling1D()(cnn2)
    cnn2 = keras.layers.Conv1D(32,3, activation='relu',padding='same', kernel_regularizer='l2')(cnn2)
    cnn2 = keras.layers.MaxPooling1D()(cnn2)
    cnn2 = keras.layers.Flatten()(cnn2)
    # dense layers for 2d inputs
    num = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num")(energy_input)
    num = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num")(num)
    num2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num2")(spectral_input)
    num2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num2")(num2)
    num3 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2', name="dense1_num3")(zcr_input)
    num3 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2_num3")(num3)
    # stack all the layers
    full = tf.keras.layers.concatenate([cnn1, cnn2, num, num2, num3], name="concat_full")
    full= keras.layers.Dense(32, activation='relu', kernel_regularizer='l2')(full)
    full = keras.layers.Dense(classes, activation='softmax', kernel_regularizer='l2')(full)
    model = keras.models.Model(inputs = [mfcc_input, chroma_input, energy_input, spectral_input, zcr_input], outputs = full, name = "ResNet34_v2")
    return model

def reshape (X):
    return np.asarray(X).astype(np.float32)

others_lst = ['Blade saw', 'Brick cutter', 'Concrete crusher', 'Coring machine', 'Electrical drill drilling', 
              'Fastening machine', 'Fastening', 'Pneumatic driver', 'Wall chaser']
electrical_drill = ['Electrical drill percussive', 'Electrical drill', 'Electrical driller']
over10 = ['Percussive over 10']
below10 = ['Handheld less 10', 'Handheld over 10']

# chroma_others, energy_others, mfcc_others, spectral_others, zcr_others = feature_combine ('\\site_audio_json_feature', others_lst, 'Others')
# chroma_electrical, energy_electrical, mfcc_electrical, spectral_electrical, zcr_electrical = feature_combine ('\\site_audio_json_feature', electrical_drill, 'Electrical drill')
# chroma_below10, energy_below10, mfcc_below10, spectral_below10, zcr_below10 = feature_combine ('\\site_audio_json_feature', below10, 'Under 10')
# chroma_over10, energy_over10, mfcc_over10, spectral_over10, zcr_over10 = feature_combine ('\\site_audio_json_feature', over10, 'Over 10')

chroma_others, energy_others, mfcc_others, spectral_others, zcr_others = feature_combine ('\\conference_audio_json_feature', ['Others'], 'Others')
chroma_electrical, energy_electrical, mfcc_electrical, spectral_electrical, zcr_electrical = feature_combine ('\\conference_audio_json_feature', ['ElectricDrill'], 'Electrical drill')
chroma_below10, energy_below10, mfcc_below10, spectral_below10, zcr_below10 = feature_combine ('\\conference_audio_json_feature', ['Under10'], 'Under 10')
chroma_over10, energy_over10, mfcc_over10, spectral_over10, zcr_over10 = feature_combine ('\\conference_audio_json_feature', ['Over10'], 'Over 10')
chroma_back, energy_back, mfcc_back, spectral_back, zcr_back = feature_combine ('\\background_audio_json_feature', ['other'], 'Others')

chroma_f = np.vstack ((chroma_over10, chroma_below10, chroma_electrical, chroma_others, chroma_back))
energy_f = np.vstack ((energy_over10, energy_below10, energy_electrical, energy_others, energy_back))
mfcc_f = np.vstack ((mfcc_over10, mfcc_below10, mfcc_electrical, mfcc_others, mfcc_back))
spectral_f = np.vstack ((spectral_over10, spectral_below10, spectral_electrical, spectral_others, spectral_back))
zcr_f = np.vstack ((zcr_over10, zcr_below10, zcr_electrical, zcr_others, zcr_back))
label = ['Over10']*chroma_over10.shape[0] + ['Below10']*chroma_below10.shape[0] + ['Electrical drill']*chroma_electrical.shape[0] + ['Others']*(chroma_others.shape[0]+chroma_back.shape[0])
lb = LabelBinarizer()
labelz = lb.fit_transform (label)

# Model training
test_model = ResNet34_v2(mfcc_f, chroma_f, energy_f, spectral_f, zcr_f)
test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                   loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
TEST_SIZE = 0.3
BATCH_SIZE = 1
EPOCHS = 50
X_train, X_test,X2_train,X2_test,X3_train,X3_test,X4_train, X4_test,X5_train, X5_test, y_train, y_test= train_test_split(mfcc_f, chroma_f, energy_f, spectral_f, zcr_f, labelz, test_size=TEST_SIZE)
history = test_model.fit(x=[X_train,X2_train,X3_train, X4_train, X5_train], y=y_train, validation_data=([X_test, X2_test,X3_test,X4_test, X5_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
test_model.save (os.getcwd() + '\\models\\conference_with_back_js_v1.h5')

# load model
# test_model = keras.saving.load_model(os.getcwd() + '\\models\\conference_with_back_js_v1.h5')

# Prediction
y_predicted = np.argmax(test_model.predict(x=[X_test,X2_test, X3_test, X4_test, X5_test]), axis=1)
y_true = np.argmax(y_test, axis=1)
label_names = np.unique(label)
confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_predicted)
sns.heatmap(confusion_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# FPR and TPR
def accuracy (cnf_matrix):
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
    print ('TN', TN)
    print ('FP', FP)
    print ('f1:', np.average(TP/(TP+0.5*(FP+FN))))

#cnf_matrix=confusion_matrix(y_true,y_predicted)
cnf_matrix = np.array([[160, 16, 29, 9], [10, 184, 10, 1], [5, 5, 769, 0], [8, 2, 9, 158]])
accuracy (cnf_matrix)

# Site audios mixed with background 
# FPR (%): [2.99647474 1.5321155  7.22689076 0.45688178]
# ['Below10', 'Electrical drill', 'Others', 'Over10']

# Conference audios mixed with background 
# FPR (%): [4.70723307 3.28947368 3.34128878 2.37324703]
# ['Below10', 'Electrical drill', 'Others', 'Over10']


