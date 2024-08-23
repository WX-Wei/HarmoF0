
import h5py
import numpy as np
from multiprocessing import Pool
import os
import torch
import torchaudio
import librosa
from tqdm import tqdm
import glob
from mono_pitch.mono_utils import split_path

dataset_dir = 'MDB-stem-synth'

wav_dir = f'data/{dataset_dir}/audio_stems'
label_dir = f'data/{dataset_dir}/annotation_stems'
dest_dir = f'data/{dataset_dir}/h5'

def prepare_dataset(wav_path):
    # resample audio and make labels.
    sample_rate = 16000
    dir_name, basename, ext = split_path(wav_path)
    label_path = os.path.join(label_dir, basename + '.csv')
    h5_path = os.path.join(dest_dir, basename + '.h5')
    dest_label_path = os.path.join(dest_dir, basename + '.txt')
    if not os.path.exists(h5_path):       
        waveform, sr = librosa.load(wav_path, sr=sample_rate)  
        label = np.loadtxt(label_path, delimiter=',') # 10ms, delimiter=' '
        with h5py.File(h5_path, 'w') as h5:
            h5['waveform'] = waveform
            h5['sample_rate'] = sample_rate
        np.savetxt(dest_label_path, label, header='time, frequency', fmt="%.3f")

if __name__ == "__main__":
    os.makedirs(dest_dir, exist_ok=True)
    wav_list = glob.glob(os.path.join(wav_dir,'*.wav'))


    for i in tqdm(wav_list):
        prepare_dataset(i)

 
    # with Pool(8) as p:
    #     res = list(tqdm(p.imap(prepare_dataset, wav_list), total=len(wav_list)))