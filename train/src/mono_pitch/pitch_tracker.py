# monophonic pitch estimator using harmonic_net.

# torch
from random import shuffle
import torch
import torch.cuda
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
import torchaudio
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from sacred.observers import FileStorageObserver


# system
from tqdm import tqdm
from glob import glob
from datetime import datetime
import os
import sys
import json

# others
from sacred.utils import SacredInterrupt

class EarlyStopInterrupt(SacredInterrupt):
    STATUS = 'Early Stop'

import librosa
import numpy as np
import mir_eval
import pandas as pd
import matplotlib.pyplot as plt
# 
from dataset_f0 import MonoPitchDataset, InferencePitchDataset
from nets import HarmoF0

from mono_config import ex
from mono_utils import plot_pitch_result, pretty_dict


class PitchTracker():
    def __init__(self, 
        net,
        ex,
        writer,
        train_set = None,
        validate_set = None,
        test_set = None,
        inference_set = None,
    ) -> None:
        self.global_step = 0
        self.net = net
        self.writer = writer
        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set
        self.inference_set = inference_set

        self.cfg = ex.current_run.config
        self.max_epochs = self.cfg['max_epochs']
        self.device = self.cfg['device']
        self.postive_weight = self.cfg['postive_weight']
        self.pitch_loss_weight = self.cfg['pitch_loss_weight']

        self.freq_bins_in = self.cfg['freq_bins_in']
        self.freq_bins_out = self.cfg['freq_bins_out']
        self.bins_per_octave_in = self.cfg['bins_per_octave_in']
        self.bins_per_octave_out = self.cfg['bins_per_octave_out']
        self.fmin = self.cfg['fmin']
        self.skip_training = self.cfg['skip_training']

        self.early_stop_score = 0
        self.early_stop_epoch = 0

    def save_checkpoint(self, epoch):
        # save checkpoint
        check_point_dir = os.path.join(self.result_dir, 'checkpoints')
        os.makedirs(check_point_dir, exist_ok=True)

        # check_point_path = check_point_dir + '/epoch_%d_checkpoint.pth'%(epoch)
        # torch.save(self.net.state_dict(), check_point_path)
        # model_path = check_point_dir + '/epoch_%d_checkpoint.all.pth'%(epoch)
        # torch.save(self.net, model_path) # model = torch.load(path)

        check_point_path = check_point_dir + '/best_checkpoint.pth'
        torch.save(self.net.state_dict(), check_point_path)
        model_path = check_point_dir + '/best_checkpoint.all.pth'
        torch.save(self.net, model_path) # model = torch.load(path)

    def fit(self):
        self.global_step = 0
        self.best_scores_dict = {}
        
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        for epoch in range(self.max_epochs):
            self.curr_scores_dict = {}
            self.curr_scores_dict['epoch'] = epoch
            print('='*80)
            # print('epoch:%d/%d'%(epoch, self.max_epochs))
            if(not self.skip_training):
                self.one_epoch(epoch, self.train_set, subset='train', optimizer=optimizer)

            self.one_epoch(epoch, self.test_set, subset='test')
            self.one_epoch(epoch, self.validate_set, subset='validate')

    def inference(self, wav_path, batch_size, n_frame):
        self.result_dir, basename = os.path.split(wav_path)
        wav_name, ext = os.path.splitext(basename)
        pred_path = os.path.join(self.result_dir, wav_name + ".pred.txt")
        activation_path = os.path.join(self.result_dir, wav_name + ".activation.png")

        waveform, sr = torchaudio.load(wav_path)
        print("dur:", waveform.shape[1]/sr, sr)
        if(sr != self.cfg['sample_rate']):
            print("convert sr from %d to %d"%(sr, self.cfg['sample_rate']))
            resampler = torchaudio.transforms.Resample(sr, self.cfg['sample_rate']).to(self.device)
            waveform = waveform.to(self.device)
            waveform = resampler(waveform)
        waveform = torch.sum(waveform, dim=0)
        waveform = waveform.cpu().numpy()
        
        # inference_dataset = InferencePitchDataset(wav_path)
        inference_dataset = InferencePitchDataset(waveform=waveform)
        print('dataset len', len(inference_dataset))
        print('batch_size, n_frame', batch_size, n_frame)
        inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size * n_frame, shuffle=False)

        # self.one_epoch(0, self.inference_set, subset='inference')
        #######################################3
        # test
        epoch = 0
        step = 0
        subset = 'inference'

        result_dict = {
            'wav_ids':[],
            'times':[],
            'true_freqs':[],
            'pred_freqs':[],
            'pred_activations':[],
            'pred_activations_map':[],
            'specgram': []
        }

        pbar = tqdm(inference_dataloader)
        pbar.set_description('epoch:%d/%d %s'%(epoch, 1, subset))
        for waveforms, true_freqs, wav_ids, times in pbar:
            step += 1

            print('waveform.shape', waveform.shape)

            n_frame = self.cfg['n_frame']
            # batch_num = true_freqs.size()[0]
            # batch_num = batch_num - (batch_num % n_frame) # 整除n_frame
            # waveforms = waveforms[:batch_num]
            # times = times[:batch_num]

            with torch.no_grad():
                # => [b x T x (88*4)]
                est_onehot, specgram = self.net.eval()(waveforms, True)

            # scores = self.score_of_one_batch(true_freqs, est_onehot, result_dict, times, wav_ids)
            est_freqs, est_activations = self.onehot_to_hz(est_onehot, self.bins_per_octave_out, threshold=0)
            est_freqs_flatten = est_freqs.flatten().cpu().detach().numpy()
            est_activations_flatten = est_activations.flatten().cpu().detach().numpy()

            times_flatten = times.flatten().cpu().detach().numpy()

            result_dict['times'] += list(times_flatten)
            result_dict['pred_freqs'] += list(est_freqs_flatten)
            result_dict['pred_activations'] += list(est_activations_flatten)
            result_dict['pred_activations_map'] += [est_onehot.flatten(0, 1).cpu().detach().numpy()]

        # # to csv
        # df = pd.DataFrame.from_dict(scores_list_dict)
        # os.makedirs(self.result_dir, exist_ok=True)
        # df.to_csv(self.result_dir + '/epoch_%d_%s_scores_list.csv'%(epoch, subset))

        # if(subset=='test'):
        #     rm_lst = glob(self.result_dir + '/epoch_*_pred_results.csv')
        #     for p in rm_lst:
        #         os.remove(p)
        #     df = pd.DataFrame.from_dict(result_dict)
        #     df.to_csv(self.result_dir + '/epoch_%d_%s_pred_results.csv'%(epoch, subset))

        # avg metrics

        times = np.array(result_dict['times'])
        true_freq = np.array(result_dict['true_freqs'])
        pred_freq = np.array(result_dict['pred_freqs'])
        pred_activation = np.array(result_dict['pred_activations'])
        pred_activation_map = np.concatenate(result_dict['pred_activations_map'], axis=0)

        print("output inference result!")
        pred_table = np.stack([times, pred_freq, pred_activation], axis=1)
        np.savetxt(pred_path, pred_table, header='time frequency activation', fmt="%.03f")
        if(self.cfg['save_activation']):
            plt.imsave(activation_path, pred_activation_map.T[::-1])
        
        
    def score_of_one_batch(self, true_freqs, est_onehot, result_dict, times, wav_ids):
        # resize a batch to 1 sample
        b_, T_ = true_freqs.size()
        true_freqs_flatten = true_freqs.reshape([b_*T_]).cpu().detach().numpy()

        est_freqs, est_activations = self.onehot_to_hz(est_onehot, self.bins_per_octave_out, threshold=0)
        est_freqs_flatten = est_freqs.reshape([b_*T_]).cpu().detach().numpy()
        est_activations_flatten = est_activations.reshape([b_*T_]).cpu().detach().numpy()

        times_flatten = times.reshape([b_*T_]).cpu().detach().numpy()
        wav_ids_flatten = wav_ids.reshape([b_*T_]).cpu().detach().numpy()

        result_dict['wav_ids'] += list(wav_ids_flatten)
        result_dict['times'] += list(times_flatten)
        result_dict['true_freqs'] += list(true_freqs_flatten)
        result_dict['pred_freqs'] += list(est_freqs_flatten)
        result_dict['pred_activations'] += list(est_activations_flatten)

        scores = mir_eval.melody.evaluate(times_flatten, true_freqs_flatten, times_flatten, est_freqs_flatten)
        return scores

    def loss(self, ref_notes, est_notes):
        pitch_loss = - self.postive_weight * ref_notes * torch.log(est_notes) - (1 - ref_notes)*torch.log(1-est_notes)
        pitch_loss = pitch_loss.mean() * self.pitch_loss_weight
        assert(torch.isinf(pitch_loss).sum() == 0)
        assert(torch.isnan(pitch_loss).sum() == 0)
        return pitch_loss

    def hz_to_onehot(self, hz, freq_bins, bins_per_octave):
        # input: [b x T]
        # output: [b x T x freq_bins]

        fmin = self.cfg['fmin']

        indexs = ( torch.log((hz+0.0000001)/fmin) / np.log(2.0**(1.0/bins_per_octave)) + 0.5 ).long()
        assert(torch.max(indexs) < freq_bins)
        mask = (indexs >= 0).long()
        # => [b x T x 1]
        mask = torch.unsqueeze(mask, dim=2)
        # => [b x T x freq_bins]
        onehot = F.one_hot(torch.clip(indexs, 0), freq_bins)
        onehot = onehot * mask # mask the freq below fmin
        return onehot

    def onehot_to_hz(self, onehot, bins_per_octave, threshold = 0.6):
        # input: [b x T x freq_bins]
        # output: [b x T]
        fmin = self.cfg['fmin']
        max_onehot = torch.max(onehot, dim=2)
        indexs = max_onehot[1]
        mask = (max_onehot[0] >= threshold).float()

        hz = fmin * (2**(indexs/bins_per_octave))
        hz = hz * mask # set freq to 0 if activate val below threshold
        
        return hz, max_onehot[0]

    def plot_est_pitch_on_piano_roll(self, true_freqs, est_onehot, specgram, wav_ids, subset, epoch, step):
        true_midis = librosa.hz_to_midi((true_freqs + 0.0001).cpu().detach().numpy())
        est_freqs, est_activations = self.onehot_to_hz(est_onehot, self.bins_per_octave_out, threshold=0.1)
        est_freqs = est_freqs.cpu().detach().numpy()
        est_freqs[est_freqs<=0] = np.nan
        est_midis = librosa.hz_to_midi(est_freqs)
        
        wav_name = 'wav_id=%d'%wav_ids[0, 0]
        fig = plot_pitch_result(true_midis[0], est_midis[0], est_note=est_onehot[0], specgram=specgram[0], wav_name=wav_name)
        if(self.writer):
            self.writer.add_figure('%s/epoch_%d_step_%d'%(subset,epoch,step), fig)
        if(self.cfg['save_fig']):
            fig.savefig(os.path.join(self.result_dir + '/epoch_%d'%epoch,  '%s_step_%d_%s.png'%(subset, step, wav_name)))

    def validate_one_epoch(self, epoch, dataloader, subset='validate'):
        self.one_epoch(epoch, dataloader, subset)

    def one_epoch(self, epoch, dataloader,  subset, optimizer=None):
        #######################################3
        # test
        os.makedirs(self.result_dir + '/epoch_%d'%epoch, exist_ok=True)
        step = 0
        scores_list_dict ={
            'wav_name':[],
            'Voicing Recall': [],
            'Voicing False Alarm': [],
            'Raw Pitch Accuracy': [],
            'Raw Chroma Accuracy': [],
            'Overall Accuracy': []
        }

        result_dict = {
            'wav_ids':[],
            'times':[],
            'true_freqs':[],
            'pred_freqs':[],
            'pred_activations':[]
        }
        hop_time = self.cfg['hop_length']/self.cfg['sample_rate']

        pbar = tqdm(dataloader)
        pbar.set_description('epoch:%d/%d %s'%(epoch, self.max_epochs, subset))
        for waveforms, true_freqs, wav_ids, times in pbar:
            step += 1

            n_frame = self.cfg['n_frame']
            batch_num = true_freqs.size()[0]
            batch_num = batch_num - (batch_num % n_frame) # 整除n_frame
            waveforms = waveforms[:batch_num]
            true_freqs = true_freqs[:batch_num]
            wav_ids = wav_ids[:batch_num]
            times = times[:batch_num]

            true_freqs = true_freqs.reshape([int(batch_num//n_frame), n_frame])
            wav_ids = wav_ids.reshape([int(batch_num//n_frame), n_frame])

            ref_onehot = self.hz_to_onehot(true_freqs, self.freq_bins_out, self.bins_per_octave_out)
            ref_onehot = ref_onehot.to(self.device)

            # train
            if(subset == 'train'):
                self.global_step += 1
                optimizer.zero_grad()
                # => [b x T x (88*4)]
                est_onehot, specgram = self.net(waveforms)
                loss = self.loss(ref_onehot, est_onehot)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                # pbar.set_postfix(loss='%.6f'%(loss),  )
                self.writer.add_scalar('train/loss', loss, global_step=self.global_step)

            # validate and test
            else:
                with torch.no_grad():
                    # => [b x T x (88*4)]
                    est_onehot, specgram = self.net.eval()(waveforms)
                    loss = self.loss(ref_onehot, est_onehot)

            scores = self.score_of_one_batch(true_freqs, est_onehot, result_dict, times, wav_ids)

            # scores_list_dict['wav_id'].append(wav_ids[0])
            # for key in scores.keys():
            #     scores_list_dict[key].append(scores[key])

            if step % 5 == 0:
                pbar.set_postfix(wav_id=wav_ids[0,0].cpu().numpy(), RPA=scores['Raw Pitch Accuracy'], loss='%.3f'%(loss))

            # plot results
            if(step < 40 and subset=='test'):
                self.plot_est_pitch_on_piano_roll(true_freqs, est_onehot, specgram, wav_ids, subset, epoch, step)

        # # to csv
        # df = pd.DataFrame.from_dict(scores_list_dict)
        # os.makedirs(self.result_dir, exist_ok=True)
        # df.to_csv(self.result_dir + '/epoch_%d_%s_scores_list.csv'%(epoch, subset))

        if(subset=='test'):
            rm_lst = glob(self.result_dir + '/epoch_*_pred_results.csv')
            for p in rm_lst:
                os.remove(p)
            df = pd.DataFrame.from_dict(result_dict)
            df.to_csv(self.result_dir + '/epoch_%d_%s_pred_results.csv'%(epoch, subset))

        # avg metrics

        times = np.array(result_dict['times'])
        true_freq = np.array(result_dict['true_freqs'])
        pred_freq = np.array(result_dict['pred_freqs'])
        pred_activation = np.array(result_dict['pred_activations'])
        avg_scores = mir_eval.melody.evaluate(times, true_freq, times, pred_freq)

        if(subset=='inference'):
            print("output inference result!")
            pred_table = np.stack([times, true_freq, pred_freq, pred_activation], axis=1)
            np.savetxt(self.result_dir + '/epoch_%d_prediction.txt'%(epoch), pred_table, header='time label pred activation', fmt="%.3f")
        
        for key, item in avg_scores.items():
            label = '%s/%s'%(subset, key)
            ex.log_scalar(label, item)
            self.writer.add_scalar(label, item, global_step=epoch)

        self.curr_scores_dict[subset] = avg_scores
        
        if(subset=='validate'):
            with open(self.result_dir + '/epoch_%d_avg_scores.json'%(epoch), 'a+') as f:
                f.write(pretty_dict(self.curr_scores_dict))
                f.write('\n')

            self.early_stop(avg_scores['Raw Pitch Accuracy'], epoch)

    def early_stop(self, score, epoch):
        
        if(self.early_stop_score < score):
            # save checkpoint
            self.save_checkpoint(epoch)
            self.early_stop_score = score
            self.early_stop_epoch = 0
            # update best scores
            self.best_scores_dict = self.curr_scores_dict
            best_path = self.result_dir + '/best_avg_scores[epoch=%d].json'%epoch
            with open(best_path, 'w') as f:
                f.write(pretty_dict(self.curr_scores_dict))
                f.write('\n')
            ex.add_artifact(best_path)
            ex.info['best_scores'] =  self.best_scores_dict
            ex.log_scalar('score', self.best_scores_dict['test']['Raw Pitch Accuracy'])
        else:
            self.early_stop_epoch += 1
            if(self.early_stop_epoch > self.cfg['early_stop_epoch']):
                print('='*50)
                print('Early stop: max_score=%f'%self.early_stop_score)
                raise EarlyStopInterrupt()
                exit()


@ex.automain
def main(
        batch_size, 
        n_frame, 
        n_freq, 
        n_har,
        hop_length,
        bins_per_octave_in,
        sample_rate,
        dataset_name,
        device,
        checkpoint_path,
        add_accompaniments,
        SNR,
        dilation_modes,
        dilation_rates,
        logspecgram_type,
        channels,
        fold_k,
        random_clip,
        fmin,
        freq_bins_in,
        frame_length,
        inference_mode,
        wav_path,
        ):

        # Load Model
        harmonic_f0 = HarmoF0(
            sample_rate=sample_rate,
            n_freq=n_freq,
            n_har=n_har,
            bins_per_octave=bins_per_octave_in,
            hop_length=hop_length,
            device=device,
            dilation_modes=dilation_modes,
            dilation_rates=dilation_rates,
            logspecgram_type = logspecgram_type,
            channels = channels,
            fmin = fmin,
            freq_bins=freq_bins_in,
            config=ex.current_run.config,
        )

        # Load checkpoint
        if(checkpoint_path):
            harmonic_f0.load_state_dict(torch.load(checkpoint_path))
        harmonic_f0 = harmonic_f0.to(device)

        net = harmonic_f0
        net.cfg = ex.current_run.config

        #####################
        # Load Dataset 
        if(inference_mode == False):
            # set file observer
            time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            ob = FileStorageObserver('results/' + time_str)
            ex.observers.append(ob)
            result_dir = ob.basedir

            # load datasets
            # dataset = MonoPitchDataset(dataset_name, frame_length=frame_length, add_accompaniments=add_accompaniments, SNR=SNR, fold_k=fold_k)
            dataset = MonoPitchDataset(dataset_name, frame_length=frame_length, add_accompaniments=add_accompaniments, SNR=SNR, fold_k=fold_k, subset='all')
            dataloader = DataLoader(dataset, batch_size=batch_size * n_frame, shuffle=False, num_workers=4)
            ex.info['dataset_train'] = dataset.wav_list

            validate_dataset = MonoPitchDataset(dataset_name, frame_length=frame_length, add_accompaniments=add_accompaniments, SNR=SNR, subset='validate', fold_k=fold_k)
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size * n_frame, shuffle=False, num_workers=4)
            ex.info['dataset_validate'] = validate_dataset.wav_list

            test_dataset = MonoPitchDataset(dataset_name, frame_length=frame_length, add_accompaniments=add_accompaniments, SNR=SNR, subset='test', fold_k=fold_k)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size * n_frame, shuffle=False, num_workers=4)
            ex.info['dataset_test'] = test_dataset.wav_list
            
            # tensorboard
            log_dir = "./log/mono_pitch/" + time_str
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            pitch_tracker = PitchTracker(net, ex, writer, dataloader, validate_dataloader, test_dataloader)
            pitch_tracker.result_dir = result_dir
            # # preview audio file
            # for one_batch in dataloader:
            #     waveforms = one_batch[0]
            #     ref_freqs = one_batch[1] # [b x T]
            #     writer.add_graph(harmonic_f0, input_to_model= waveforms)
            #     # preview audio file
            #     for i in range(batch_size):
            #         writer.add_audio('train/%d'%(i), waveforms[i], global_step=0, sample_rate=sample_rate)
                
            #     # preview specgram
            #     # => [b x T x freq_bins]
            #     specgrams = harmonic_f0.waveform_to_logspecgram(waveforms.to(device))

            #     # => [b x T x freq_bins]
            #     ref_freqs = ref_freqs.to(device)
            #     onehot = pitch_tracker.hz_to_onehot(ref_freqs, freq_bins_in, bins_per_octave_in)
            #     specgrams = specgrams * (1-onehot)
            #     specgram_preview = specgrams.permute([0, 2, 1])
            #     specgram_preview = specgram_preview.cpu().detach().numpy()
            #     specgram_preview = specgram_preview[:, ::-1, :]
            #     for i in range(specgram_preview.shape[0]):
            #         fig = plt.figure(figsize=[5, 15])
            #         ax = plt.imshow(specgram_preview[i, :, :])
            #         fig.colorbar(ax)
            #         writer.add_figure('specgrams_preview/%d'%i, fig)

            #     break

            # module structure
            os.makedirs(result_dir, exist_ok=True)
            with open(result_dir + '/module_structure.txt', 'w') as f:
                parm_num = sum([param.nelement() for param in net.parameters()])
                f.write('total parameter: %.3fM\n\n' % (parm_num/1e6))
                f.write(str(net))
            ex.info['model parameters'] = '%.3fM'%(parm_num/1e6)
            ex.add_artifact(result_dir + '/module_structure.txt')

            pitch_tracker.fit()

        else:
            pitch_tracker = PitchTracker(net, ex, writer = None)
            pitch_tracker.inference(wav_path, batch_size, n_frame)