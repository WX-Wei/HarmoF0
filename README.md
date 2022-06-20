# HarmoF0 Pitch Tracker

This repo is the Pytorch implementation of ["HarmoF0: Logarithmic Scale Dilated Convolution For Pitch Estimation"](https://arxiv.org/abs/2205.01019). 

HarmoF0 is a light-weight and high-performance pitch tracking model using multi-rate dilated convolution. The evaluation results with threshold of 50 cents are:
  
![ ](https://github.com/WX-Wei/HarmoF0/raw/86ed681d34cecab106309af7eff23a5691c0c85a/img/table2.png)

## Installing and usage

Install required packages and harmof0 using

```
$ pip install -r requirements.txt
$ python setup.py install
```

Estimating pitch of an audio file or folder using pretrained harmof0. HarmoF0 supports wav, mp3 and flac.

```
$ harmof0 test/a.mp3 
$ harmof0 test
```

The results are saved in test/a.f0.txt and test/a.activation.png by default.





a.f0.txt
```
# time frequency activation
0.000 27.500 0.000
0.010 27.500 0.000
0.020 27.500 0.000
0.030 27.500 0.000
0.040 27.500 0.000
0.050 27.500 0.000
0.060 27.500 0.000
0.070 27.500 0.000
0.080 55.000 0.047
0.090 55.000 0.032
0.100 59.118 0.062
```

a.activation.png

![ ](https://github.com/WX-Wei/HarmoF0/raw/86ed681d34cecab106309af7eff23a5691c0c85a/img/a.activation.png)


Use specified output dir and device:

```
$ harmof0 test/a.mp3 --output-dir=output --device=cuda
```

For more information:

```
$ harmof0 --help
```

## Usage inside Python

Import harmof0 as module:

```python
import harmof0
import torchaudio

pit = harmof0.PitchTracker()
waveform, sr = torchaudio.load('test/a.mp3')
time, freq, activation, activation_map = pit.pred(waveform, sr)

```








