import librosa
import numpy as np
import matplotlib.pyplot as plt
import json

import torch.nn.functional as F
import torch

import os
from posixpath import basename
from shutil import copyfile

def split_path(file_path):
    dir_path, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    return dir_path, base_name, ext

def pretty_dict(d):
    return json.dumps(d, indent=4, ensure_ascii=False)

def to_freq(est_notes, true_midis, hop_time, threshold = 0.6):
    est_notes_max = torch.max(est_notes, dim=2)
    est_notes_mask = est_notes_max[0] < threshold
    
    pred_midis = ((est_notes_max[1] + 21 * 4 - 2)/4).float()
    # pred_midis = ((est_notes_max[1] + 21 * 4)/4).float()
    pred_midis = torch.masked_fill(pred_midis, est_notes_mask, value=0)
    
    true_midis = true_midis.numpy()[0].astype(np.float)
    pred_midis = pred_midis.cpu().numpy()[0]

    # 
    true_time = np.array([hop_time * (i + 1) for i in range(len(true_midis))])
    pred_time = np.array([hop_time * (i + 1) for i in range(len(pred_midis))])

    true_freq = librosa.midi_to_hz(true_midis).astype(np.float32)
    pred_freq = librosa.midi_to_hz(pred_midis)

    true_freq[true_freq <= 10.0] = 0
    pred_freq[pred_freq <= 10.0] = 0

    return true_time, true_freq, pred_time, pred_freq, pred_midis


def plot_pitch_result(true_midi, pred_midi, est_note, specgram, wav_name):
    # true_midis: []
    fig, axs = plt.subplots(3, 1)
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    title = 'pred_pitch %s'%(wav_name)
    ax1.set_title(title)
    true_midi[true_midi <= 0.0] = np.NaN
    pred_midi[pred_midi <= 0.0] = np.NaN
    ax1.plot(true_midi)
    ax1.plot(pred_midi)
    ax1.legend(['true', 'pred'])
    ax1.set_xlim([0, len(true_midi)])
    ax1.get_xaxis().set_visible(False)

    # => [88 x T]
    est_notes_img = est_note.permute([1, 0])
    est_notes_img = est_notes_img.cpu().detach().numpy()[::-1]
    # ax2.set_title('pred notes')
    ax2.get_xaxis().set_visible(False)
    ax2_ = ax2.imshow(est_notes_img, aspect='auto')
    fig.colorbar(ax2_, ax = ax2, fraction=0.05, pad=0.05)

    fig.colorbar(ax2_, ax = ax1, fraction=0.05, pad=0.05)

    #fig.colorbar(ax2_, ax = ax3, orientation='horizontal') # 'vertical'

    specgram_img = specgram.permute([1, 0])
    specgram_img = specgram_img.cpu().detach().numpy()[::-1]
    ax3_ = ax3.imshow(specgram_img, aspect='auto')
    fig.colorbar(ax3_, ax = ax3, fraction=0.05, pad=0.05)
    
    return fig


def getWeightedAvg(dataframe, weight_col, cols):
    weighted_avg_dict = {}
    total_weights = dataframe[weight_col].sum()
    weights = dataframe[weight_col] / total_weights

    for c in cols:
        weighted_avg_dict[c] = (dataframe[c] * weights).sum()

    return weighted_avg_dict



def tb_add_notes_img(x, writer, global_step, tag):
    # inputs: [b x T x n_fre]
    b, T, n_fre = x.size()

    x = x.permute([0, 2, 1])
    x = x.reshape([b, 1, n_fre, T])
    x = torch.cat([x * 1, x , x * 0.5], dim=1)
    max_val = torch.max(x)
    # x = F.pad(x, pad=[1,1,1,1], value=max_val)
    
    x = x.cpu().detach().numpy()
    x = x[:, :, ::-1]
    writer.add_images('step_%d_'%(global_step) + tag, x, global_step=global_step)