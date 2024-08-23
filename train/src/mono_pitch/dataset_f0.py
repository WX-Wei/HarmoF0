
import numpy as np
import h5py
from multiprocessing import Pool

from torch.utils import data
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
import torchaudio
p_path = os.path.split(__file__)[0] + '/../'
sys.path.append(p_path)

from tqdm import tqdm
import glob
import librosa

from sacred import Experiment

os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

def add_noise(x, noise, snr):
    """
    :param x: signal
    :param snr: signal to noise rate
    :return: signal
    """
    P_signal = np.mean(x**2)    
    P_noise = np.mean(noise**2)

    # k = np.sqrt(P_signal / 10 ** (snr / 10.0))  #
    k = np.sqrt(P_signal/P_noise / 10 ** (snr / 10.0))  #

    # return x + noise

    return x + noise * k

class SingleWavDataset(Dataset):
    def __init__(self, 
            wav_path,
            file_id,
            add_accompaniments=False, 
            SNR=0,
            fixed_len = 0,
            frame_length = 1024
        ) -> None:
        super().__init__()

        self.wav_path = wav_path
        self.add_accompaniments = add_accompaniments
        self.SNR = SNR
        self.fixed_len = fixed_len
        self.frame_length = frame_length
        self.file_id = file_id

        with h5py.File(wav_path, mode='r') as h5:
            self.sample_rate = h5['sample_rate'][()]
            self.wav_len = h5['waveform'].shape[0]

        self.wav_name = os.path.basename(wav_path)
        labels = np.loadtxt(wav_path[:-3] + ".txt")
        self.times = labels[:, 0]
        self.freqs = labels[:, 1]
        label_len = labels.shape[0]
        if(fixed_len > 0):
            self.dataset_len = min(fixed_len, label_len)
        else:
            self.dataset_len = label_len

    def __getitem__(self, index):
        # output: 
        time = self.times[index]
        freq = self.freqs[index]

        # waveform
        begin = int(self.sample_rate * time - self.frame_length//2)

        if(begin >= self.wav_len - self.frame_length):
            begin = self.wav_len - self.frame_length - 1

        end = begin + self.frame_length
        offset = 0
        if(begin < 0):
            offset = - begin
            begin = 0
        pad = np.zeros(offset)
        with h5py.File(self.wav_path, mode='r') as h5:
            waveform = h5['waveform'][begin:end]
        waveform = np.append(pad, waveform)
        if(self.add_accompaniments):
            assert False, "not implemented yet."
            # waveform = add_noise(waveform, accompaniments, self.SNR)
                
        return waveform, freq, self.file_id, time

    def __len__(self):
        return self.dataset_len


class MonoPitchDataset(Dataset):
    def __init__(self,
            dataset_name='MIR-1K',
            frame_length = 1024,
            fixed_len = 0,
            add_accompaniments=False, 
            SNR=0,
            subset = 'train',
            fold_k = 0,
        ) -> None:
        super().__init__()

        wav_list = glob.glob('data/' + dataset_name + '/h5/*.h5')
        wav_list.sort()

        # split into training set and test set.
        test_list = wav_list[fold_k:len(wav_list):5]
        validate_list = wav_list[fold_k+1:len(wav_list):5]
        train_list = [x for x in wav_list if x not in test_list and x not in validate_list]

        if(subset=='test'):
            self.wav_list = test_list
        elif(subset=='train'):
            self.wav_list = train_list
        elif(subset=='validate'):
            self.wav_list = validate_list
        elif(subset=='all'):
            self.wav_list = wav_list

        self.dataset_list = []
        for i in tqdm(range(len(self.wav_list)), desc='processing wav list'):
            wav_path = self.wav_list[i]
            self.dataset_list += [SingleWavDataset(wav_path, i, add_accompaniments, SNR, frame_length=frame_length, fixed_len=fixed_len)]

        print('\n\n%s List len: %d\n'%(dataset_name,len(self.dataset_list)))
        self.concat_dataset = ConcatDataset(self.dataset_list)

        self.dataset_size = len(self.concat_dataset)


    def __getitem__(self, index) :
        return self.concat_dataset[index]
    def __len__(self):
        return self.dataset_size


class InferencePitchDataset(Dataset):
    def __init__(self,
            waveform = None,
            wav_path=None,
            frame_length = 1024,
            hop_length = 320,
            sample_rate = 16000,
        ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.file_id = 0

        if(not waveform is None):
            self.waveform = waveform
        else:
            self.waveform, _ = librosa.load(wav_path, sr=sample_rate)

        self.subset = 'inference'

        self.wav_len = len(self.waveform)
        self.dataset_size = int(self.wav_len / hop_length)

        hop_time = hop_length / sample_rate

        self.times = np.arange(self.dataset_size) * hop_time
        self.freqs = np.zeros(self.dataset_size)

    def __getitem__(self, index) :

        time = self.times[index]
        freq = self.freqs[index]

        # waveform
        begin = int(self.sample_rate * time - self.frame_length//2)

        if(begin >= self.wav_len - self.frame_length):
            begin = self.wav_len - self.frame_length - 1

        end = begin + self.frame_length
        offset = 0
        if(begin < 0):
            offset = - begin
            begin = 0
        pad = np.zeros(offset)
        waveform = self.waveform[begin:end]
        waveform = np.append(pad, waveform)

        return waveform, freq, self.file_id, time

    def __len__(self):
        return self.dataset_size

ex = Experiment()

@ex.automain
def main():
    s = MonoPitchDataset(dataset_name='MIR-1K')
    
    for waveform, freq, file_id, time in s:
        x = waveform
    pass