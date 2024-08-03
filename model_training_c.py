import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
# import tensorflow as tf
# from tensorflow import keras
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, det_curve, ConfusionMatrixDisplay
import pickle
import seaborn as sns

def load_pme ():
    features = pickle.load(open(os.getcwd() + '\\PME_conference_features\\mel_list_app.pkl', 'rb')).tolist()
    zcr_list = pickle.load(open(os.getcwd() + '\\PME_conference_features\\zcr_list_app.pkl', 'rb')).tolist()
    cent_list = pickle.load(open(os.getcwd() + '\\PME_conference_features\\cent_list_app.pkl', 'rb')).tolist()
    label = pickle.load(open(os.getcwd() + '\\PME_conference_features\\label_list_app.pkl', 'rb'))
    cens_list = pickle.load(open(os.getcwd() + '\\PME_conference_features\\cens_list_app.pkl', 'rb')).tolist()
    length = len(label)
    features = np.array(features).reshape (length,128,16,6)
    zcr_list = np.array(zcr_list).reshape (length,44)
    cent_list = np.array(cent_list).reshape (length,44)
    cens_list = np.array(cens_list).reshape (length, 12*44)
    return features, zcr_list, cent_list, cens_list, label

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
feature_pme, zcr_pme, cent_pme, cens_pme, label_pme = load_pme ()
feature_back, zcr_back, cent_back, cens_back, label_back = load_back ()

label = label_pme + label_back
lb = LabelBinarizer()
labelz = lb.fit_transform (label)

# stack features
features = np.vstack ((feature_pme, feature_back))
zcr_list = np.vstack ((zcr_pme, zcr_back))
cent_list = np.vstack ((cent_pme, cent_back))
cens_list = np.vstack ((cens_pme, cens_back))
#labelz = np.vstack ((labelz_pme, labelz_back))

# build up the model
test_model = ResNet34_v2(features, cens_list, cent_list, zcr_list)
test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
TEST_SIZE = 0.3
BATCH_SIZE = 1
EPOCHS = 50
X_train, X_test,X2_train,X2_test,X3_train,X3_test,X4_train, X4_test,y_train, y_test = train_test_split(features,cens_list,cent_list,zcr_list,labelz, test_size=TEST_SIZE)
history = test_model.fit(x=[X_train,X2_train,X3_train, X4_train], y=y_train, validation_data=([X_test, X2_test,X3_test,X4_test], y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
test_model.save ('resnet_v2_test_v0.h5')
#test_model = tf.keras.models.load_model('test.h5')

#Confusion Matrix for trained model
y_predicted = np.argmax(test_model.predict(x=[X_test,X2_test, X3_test, X4_test]), axis=1)
y_true = np.argmax(y_test, axis=1)
label_names = np.unique(label)
confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_predicted)
sns.heatmap(confusion_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Further train over 10kg and under10kg, only need the pme audio files
def find(ls, pme_tool):
    index_ = [i for i, x in enumerate(ls) if x == pme_tool]
    feature_ = [feature_pme[i] for i in index_]
    #zcr_ = [zcr_pme[i] for i in index_]
    #cent_ = [cent_pme[i] for i in index_]
    cens_ = [cens_pme[i] for i in index_]
    label_ = [label_pme[i] for i in index_]
    return np.array(feature_), np.array(cens_), label_

def svc_ (x, y):
  svc = SVC()
  parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'gamma':('scale', 'auto'), 'C': [0.01, 0.1, 1, 3, 5, 10]}
  clf = GridSearchCV (svc, parameters)
  clf.fit (x, y)
  print ('Best parameters:',clf.best_params_, 'Score:',clf.score (x, y), 'Best estimator:', clf.best_estimator_)
  return clf.best_estimator_

feature_pme, zcr_pme, cent_pme, cens_pme, label_pme = load_pme ()
feature_over10, cens_over10, label_over10 = find(label_pme, 'Handheld over 10')
feature_under10, cens_under10, label_under10 = find(label_pme, 'Handheld less 10')

log_ = np.vstack ((feature_over10, feature_under10))
label_log = ["Handheld percussive breakers >10kg"] * len (feature_over10) + ["Handheld percussive breakers <10kg"] * len(feature_under10)
log_train, log_test, log_y_train, log_y_test = train_test_split(log_.reshape (637, 128*16*6), label_log, test_size=0.33, random_state=42)
log_svc = svc_ (log_train, log_y_train)

cm = confusion_matrix(log_y_train, log_svc.predict (log_train), labels=log_svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_svc.classes_)
fig, ax = plt.subplots(figsize=(14,10))
disp.plot(ax = ax)
plt.title ('(a) Train data')
plt.show()
cm = confusion_matrix(log_y_test, log_svc.predict (log_test), labels=log_svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_svc.classes_)
fig, ax = plt.subplots(figsize=(14,10))
disp.plot(ax = ax)
plt.title ('(b) Test data')
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

cnf_matrix=confusion_matrix(log_y_test, log_svc.predict (log_test))
#cnf_matrix = np.array([[160, 16, 29, 0], [10, 184, 10, 1], [5, 5, 769, 0], [8, 2, 9, 158]])
accuracy (cnf_matrix)