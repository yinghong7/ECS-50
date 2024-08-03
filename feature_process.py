from pydub import AudioSegment
import librosa
import os, glob
import numpy as np
import pickle
import torch
from pydub import AudioSegment
import pandas as pd

AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"
AudioSegment.ffprobe = os.getcwd()+ "\\ffprobe.exe"


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
    return mels_out,zcritem,centitem, cens

def select_data (): #urban noise not to include
    df = pd.read_csv (os.getcwd() + "\\meta\\esc50.csv")
    sub_df = df[~df['category'].isin(['helicopter', 'chainsaw', 'engine', 'train', 'hand_saw'])]
    return sub_df['filename'].to_list()


def main ():
    mel_list_, zcr_list_, cent_list_, cens_list, label_list = [], [], [], [], []
    folderpath = os.getcwd() + "\\audio"
    filename = select_data ()
    for i in filename:
        wav_filename = os.getcwd() + "\\audio\\" + i
        mel, zcr, cent, cens = preprocess (wav_filename)
        mel_list_.append (mel) #Important 
        zcr_list_.append (zcr) #Important 
        cent_list_.append (cent) #Important 
        cens_list.append (cens)
    
    mel_list_ = torch.tensor(mel_list_)
    zcr_list_ = torch.tensor(zcr_list_)
    cent_list_ = torch.tensor(cent_list_)
    cens_list = torch.tensor (cens_list)

    with open('mel_list_esc.pkl', 'wb') as file: 
        pickle.dump(mel_list_, file) 
    with open('zcr_list_esc.pkl', 'wb') as file: 
        pickle.dump(zcr_list_, file) 
    with open('cent_list_esc.pkl', 'wb') as file: 
        pickle.dump(cent_list_, file) 
    with open('cens_list_esc.pkl', 'wb') as file: 
        pickle.dump(cens_list, file) 
    


if __name__ == "__main__":
    main()
