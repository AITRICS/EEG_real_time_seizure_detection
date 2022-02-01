# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyedflib import highlevel
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal as sci_sig
from scipy.spatial.distance import pdist
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks
import math
import os
import argparse
import glob
import pickle
import numpy as np


################################################# feature extraction part ###################################################
def fractal_feature(signal, signal_samplerate, feature_samplerate):
    # this must be done after amplitude normalization
    # feature samplerate must be smaller than signalsamplerate
    result = []
    a = int(signal_samplerate/float(feature_samplerate))
    # endpoint = int(intv*math.floor(len(signal)/float(signal_samplerate)))
    for start_point in range(0, len(signal), signal_samplerate):
        # start_point = i*signal_samplerate
        onesec_signal = list(signal[start_point:start_point+signal_samplerate])

        for start_point in range(0, len(onesec_signal), a):
            one_signal = list(onesec_signal[start_point:start_point+a])
            e = 0.00000001
            timeintv = 1/float(signal_samplerate)
            signalsize = len(one_signal)
            oldpoint = [0, 0]
            time = 0; length = 0

            for k in range(signalsize):
                time += timeintv
                newpoint = [time, one_signal[k]]
                d = pdist([oldpoint, newpoint], metric='euclidean')
                length += d
                oldpoint = newpoint
            Nprime = 1/(2*e)
            
            result.append(1 + math.log(length, 2)/math.log(2*Nprime, 2))
    return np.asarray(result)


def power_spectral_density_feature(signal, samplerate, new_length):

    freq_resolution = 2 # higher the better resolution but time consuming...
    def psd(amp, begin, end, freq_resol = freq_resolution):
        return np.average(amp[begin*freq_resol:end*freq_resol], axis=0)

    nperseg = 8
    noverlap = 4
    # noverlap = 0
    nfft = samplerate * freq_resolution
    freqs, times, spec = stft(signal, samplerate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=True, boundary='zeros')
    amp = (np.log(np.abs(spec) + 1e-10))

    new_length = int(new_length)
    if abs(amp.shape[1] - new_length) > 1:
        print("Difference is huge {} {}".format(amp.shape[1], new_length))
    amp = amp[:,:new_length]

    # new_sig = sci_sig.resample(signal, len(times))
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(signal)    
    # plt.subplot(3, 1, 2)
    # plt.plot(new_sig)    
    # plt.subplot(3, 1, 3)
    # plt.pcolormesh(amp)
    # plt.show()

    # print("freqs: ", freqs)
    # print("times: ", times)
    # print("freqs: ", len(freqs))
    # print("times: ", len(times))
    # print("signal len: ", len(signal))
    # exit(1)

    psds = []

    if samplerate == 256:
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.641.3620&rep=rep1&type=pdf
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8910555
        psd1 = psd(amp,0,4)
        psd2 = psd(amp,4,8)
        psd3 = psd(amp,8,13)
        psd4 = psd(amp,13,20)
        psd5 = psd(amp,20,30)
        psd6 = psd(amp,30,40)
        psd7 = psd(amp,40,60)
        psd8 = psd(amp,60,80)
        psd9 = psd(amp,80,100)
        psd10 = psd(amp,100,128)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10]
    elif samplerate == 1024:
        psd1 = psd(amp,0,4)
        psd2 = psd(amp,4,8)
        psd3 = psd(amp,8,13)
        psd4 = psd(amp,13,20)
        psd5 = psd(amp,20,30)
        psd6 = psd(amp,30,40)
        psd7 = psd(amp,40,60)
        psd8 = psd(amp,60,80)
        psd9 = psd(amp,80,100)
        psd10 = psd(amp,100,128)
        psd11 = psd(amp,128,256)
        psd12 = psd(amp,256,512)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10, psd11, psd12]
    elif samplerate == 200:
        psd1 = psd(amp,0,4)
        psd2 = psd(amp,4,8)
        psd3 = psd(amp,8,13)
        psd4 = psd(amp,13,20)
        psd5 = psd(amp,20,30)
        psd6 = psd(amp,30,40)
        psd7 = psd(amp,40,50)
        psd8 = psd(amp,50,64)
        psd9 = psd(amp,64,80)
        psd10 = psd(amp,80,100)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10]
    else:
        print("Select correct sample rate!")
        exit(1)
    
    return psds

def spectrogram_feature(signal, samplerate, feature_samplerate):
    freq_resolution = 2 # higher the better resolution but time consuming...
    nperseg = 8
    noverlap = 150  #0.75 s
    nfft = int(samplerate * freq_resolution)
    freqs, times, spec = stft(signal, samplerate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=False, boundary=None)
    amp = (np.log(np.abs(spec) + 1e-10))
    return freqs, times, amp

def hilbert_envelope_feature(signal, n):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    # result = hilbert(signal, N=n)
    return hilbert(signal, N=n)

# def homomorphic_envelope_feature(signal, samplerate):
#     B_low, A_low = butter(1, 16/samplerate, btype='low')
#     print(B_low)
#     print(A_low)
#     return filtfilt(B_low, A_low, math.log(abs(hilbert_envelope_feature(signal, samplerate, 512)), 2))    

def butter_lowpass(lowcut, fs, order):
    nyq = 0.5*fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def butter_highpass(highcut, fs, order):
    nyq = 0.5*fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order)
    temp = np.array(data)
    temp = temp.astype(np.float)
    y = filtfilt(b, a, temp)

    b2, a2 = butter_highpass(highcut, fs, order)
    temp2 = np.array(y)
    temp2 = temp2.astype(np.float)
    y2 = filtfilt(b2, a2, temp2)
    return y2


############################################## end of feature extraction ################################################

def preprocess_eeg_data(data_path, data_type):
    print("EEG preprocess begins...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if data_type == "edf":
        sample = data[7]
        print("Label: ", sample['LABEL'])
        one_eeg = sample['DATA'][6]
        print("len of one_eeg: ", len(one_eeg))
        print("seconds of one_eeg: ", len(one_eeg)/256)
        x_time = np.linspace(0, int(len(one_eeg)/256), len(one_eeg))
        
        one_fractal = fractal_feature(one_eeg, 256, 4)
        psds = power_spectral_density_feature(one_eeg, 256)
        freqs, times, spec = spectrogram_feature(one_eeg, 256)
        hilbert1 = hilbert_envelope_feature(one_eeg, 256, 100)
        hilbert2 = hilbert_envelope_feature(one_eeg, 256, 400)
        hilbert3 = hilbert_envelope_feature(one_eeg, 256, 700)

        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(x_time, one_eeg)        
        plt.title('Original Signal (P3-REF)')
        plt.subplot(6, 1, 2)
        plt.plot(one_fractal)
        plt.title("fractal feature")
        plt.subplot(6, 1, 3)
        plt.pcolormesh(spec)
        plt.title("spectrogram feature")
        plt.subplot(6, 1, 4)
        plt.plot(hilbert1)
        plt.title("hilbert feature 1")
        plt.subplot(6, 1, 5)
        plt.plot(hilbert2)
        plt.title("hilbert feature 2")
        plt.subplot(6, 1, 6)
        plt.plot(hilbert3)
        plt.title("hilbert feature 3")
        plt.show()
        
        plt.figure()
        plt.subplot(11, 1, 1)
        plt.plot(x_time, one_eeg)      
        plt.title('Original Signal (P3-REF)')  
        plt.subplot(11, 1, 2)
        plt.plot(psds[0])
        plt.title('PSD 0~4 Hz feature')  
        plt.subplot(11, 1, 3)
        plt.plot(psds[1])
        plt.title('PSD 4~7 Hz feature')  
        plt.subplot(11, 1, 4)
        plt.plot(psds[2])
        plt.title('PSD 7~13 Hz feature')  
        plt.subplot(11, 1, 5)
        plt.plot(psds[3])
        plt.title('PSD 13~15 Hz feature')  
        plt.subplot(11, 1, 6)
        plt.plot(psds[4])
        plt.title('PSD 15~30 Hz feature')  
        plt.subplot(11, 1, 7)
        plt.plot(psds[5])
        plt.title('PSD 30~45 Hz feature')  
        plt.subplot(11, 1, 8)
        plt.plot(psds[6])
        plt.title('PSD 45~60 Hz feature')  
        plt.subplot(11, 1, 9)
        plt.plot(psds[7])
        plt.title('PSD 60~75 Hz feature')  
        plt.subplot(11, 1, 10)
        plt.plot(psds[8])
        plt.title('PSD 75~97 Hz feature')  
        plt.subplot(11, 1, 11)
        plt.plot(psds[9])
        plt.title('PSD 97~128 Hz feature')  
        plt.show()


if __name__ == '__main__':

    preprocess_eeg_data('/nfs/banner/ext01/destin/EEG_project/data/256Hz_edfwise_tse_data_from_tuh.pkl', 'edf')