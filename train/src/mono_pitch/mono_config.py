
import torch
from sacred import Experiment

# sacred config

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ex = Experiment('harmof0')

@ex.config
def experiment_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'HarmonicF0'

@ex.config
def input_config():
    # input config

    dataset_name = 'MDB-stem-synth' # 'MIR-1K' # 'MDB-stem-synth', 'PTDB-TUG'

    sample_rate = 16000
    hop_length = 320 # 160, 320, 441
    logspecgram_type = 'logharmgram' # 'cqt', 'logspecgram',

    n_har = 12 # harmonic series number
    n_freq = 512
    frame_length = n_freq*2 # fft window size
    n_frame = 100 # number of frames

    fmin = 27.5
    bins_per_octave_out = 12 * 4
    freq_bins_out = 88 * 4

    bins_per_octave_in = bins_per_octave_out
    freq_bins_in = freq_bins_out
    dilation_rates = [bins_per_octave_in] * 4

    dil_kernel_sizes = [[1, 3], [1,3], [1,3], [1,3]]

    add_accompaniments = False
    SNR = 0 # signal-to-noise ratio
    random_clip = False

@ex.config
def model_structure():
    # model structure
    dilation_modes = ['log_scale', 'fixed', 'fixed', 'fixed']
    channels = [32, 64, 128, 128]  # [32, 64, 64, 64]

@ex.config
def training_config():
    # training config
    skip_training = False
    max_epochs = 40
    batch_size = 24

    postive_weight = 20 # postive  weight of cross entropy loss
    pitch_loss_weight = 1000
    fold_k = 0 # 0, 1, 2, 3, 4 ( 5-fold cross-validation )

    checkpoint_path = None
    test_only = False
    save_fig = True

    early_stop_epoch = 10

@ex.config
def evulation_config():
    thresholds = [0.01, 0.1, 0.3, 0.5]


@ex.config
def inference_config():
    inference_mode = False
    # inference_mode = True
    save_activation = False
    wav_path = None
    # checkpoint_path = 'checkpoints/checkpoint_mir-1k.pth'

############################################################################
# Other Algorithm

@ex.named_config
def deep_f0():
    model_name = 'DeepF0'
    batch_size=12
    fmin = 32.7

    bins_per_octave_out = 12 * 5
    freq_bins_out = 360
    bins_per_octave_in = bins_per_octave_out
    freq_bins_in = freq_bins_out

@ex.named_config
def swipe():
    model_name = "SWIPE"
    batch_size = 24
    skip_training = True
    fmin = 32.7

    bins_per_octave_out = 12 * 5
    freq_bins_out = 360
    bins_per_octave_in = bins_per_octave_out
    freq_bins_in = freq_bins_out
    

