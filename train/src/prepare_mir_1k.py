
import numpy as np
import h5py
from multiprocessing import Pool
import os
import torch
import torchaudio
import librosa
from tqdm import tqdm
import glob
from mono_pitch.mono_utils import split_path

dataset_dir = 'MIR-1K'

wav_dir = f'data/{dataset_dir}/Wavfile'
label_dir = f'data/{dataset_dir}/PitchLabel'
dest_dir = f'data/{dataset_dir}/h5'

def prepare_dataset(wav_path):
    # resample audio and make labels.
    sample_rate = 16000
    dir_name, basename, ext = split_path(wav_path)
    label_path = os.path.join(label_dir, basename + '.pv')
    h5_path = os.path.join(dest_dir, basename + '.h5')
    dest_label_path = os.path.join(dest_dir, basename + '.txt')
    if not os.path.exists(h5_path):       
        waveforms, sr = librosa.load(wav_path, sr=sample_rate, mono=False)  

        label = np.loadtxt(label_path) # 20ms, delimiter=' '
        freq = librosa.midi_to_hz(label) * (label > 0 ).astype(int)
        freq_2 = np.zeros([len(freq)*2,])
        for i in range(0, len(freq) - 1):
            freq_2[i*2] = freq[i]
            freq_2[i*2+1] = (freq[i] + freq[i+1] ) / 2 * (freq[i] > 0 and freq[i+1]>0).astype(int)

        time_list = np.arange(len(freq_2)) * 0.01
        label = np.stack([time_list, freq_2], axis=1)
        with h5py.File(h5_path, 'w') as h5:
            h5['waveform'] = waveforms[1] # in mir-1k, channel 0 is accompaniments, channel 1 is singing voice
            h5['sample_rate'] = sample_rate

        np.savetxt(dest_label_path, label, header='time, frequency', fmt="%.3f")

if __name__ == "__main__":
    os.makedirs(dest_dir, exist_ok=True)
    wav_list = glob.glob(os.path.join(wav_dir,'*.wav'))

    for i in tqdm(wav_list):
        prepare_dataset(i)

    # with Pool(8) as p:
    #     res = list(tqdm(p.imap(prepare_dataset, wav_list), total=len(wav_list)))