# Copyright (c) 2022, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from scipy import signal as sci_sig


class PSD_FEATURE1(nn.Module):
    def __init__(self,
            sample_rate: int = 200,
            frame_length: int = 16,
            frame_shift: int = 8,
            feature_extract_by: str = 'kaldi'):
        super(PSD_FEATURE1, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1

        if self.feature_extract_by == 'kaldi':
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            import torchaudio

            self.transforms = torchaudio.transforms.Spectrogram(n_fft=self.freq_resolution*self.sample_rate,
                                                                win_length=frame_length,
                                                                hop_length=frame_shift)

        else:
            self.n_fft = self.freq_resolution*self.sample_rate
            self.hop_length = frame_shift
            self.frame_length = frame_length
        
    def psd(self, amp, begin, end):
        return torch.mean(amp[begin*self.freq_resolution:end*self.freq_resolution], 0)
        
    def forward(self, batch):
        psds_batch = []

        for signals in batch:
            psd_sample = []
            for signal in signals:
                if self.feature_extract_by == 'kaldi':
                    stft = self.transforms(signal)
                    amp = (torch.log(torch.abs(stft) + 1e-10))
                    
                else:
                    stft = torch.stft(
                        signal, self.n_fft, hop_length=self.hop_length,
                        win_length=self.frame_length, window=torch.hamming_window(self.frame_length),
                        center=False, normalized=False, onesided=True
                    )
                    amp = (torch.log(torch.abs(stft) + 1e-10))
                # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.641.3620&rep=rep1&type=pdf
                psd1 = self.psd(amp,0,4)
                psd2 = self.psd(amp,4,7)
                psd3 = self.psd(amp,7,13)
                psd4 = self.psd(amp,13,15)
                psd5 = self.psd(amp,14,30)
                psd6 = self.psd(amp,31,45)
                psd7 = self.psd(amp,55,100)
                
                psds = torch.stack((psd1, psd2, psd3, psd4, psd5, psd6, psd7))
                psd_sample.append(psds)

            psds_batch.append(torch.stack(psd_sample))

        return torch.stack(psds_batch)


class PSD_FEATURE2(nn.Module):
    def __init__(self,
            sample_rate: int = 200,
            frame_length: int = 16,
            frame_shift: int = 8,
            feature_extract_by: str = 'kaldi'):
        super(PSD_FEATURE2, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1

        if self.feature_extract_by == 'kaldi':
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            import torchaudio

            self.transforms = torchaudio.transforms.Spectrogram(n_fft=self.freq_resolution*self.sample_rate,
                                                                win_length=frame_length,
                                                                hop_length=frame_shift)

        else:
            self.n_fft = self.freq_resolution*self.sample_rate
            self.hop_length = frame_shift
            self.frame_length = frame_length
        
    def psd(self, amp, begin, end):
        return torch.mean(amp[begin*self.freq_resolution:end*self.freq_resolution], 0)
        
    def forward(self, batch):
        psds_batch = []

        for signals in batch:
            psd_sample = []
            for signal in signals:
                if self.feature_extract_by == 'kaldi':
                    stft = self.transforms(signal)
                    amp = (torch.log(torch.abs(stft) + 1e-10))
                    
                else:
                    stft = torch.stft(
                        signal, self.n_fft, hop_length=self.hop_length,
                        win_length=self.frame_length, window=torch.hamming_window(self.frame_length),
                        center=False, normalized=False, onesided=True
                    )
                    amp = (torch.log(torch.abs(stft) + 1e-10))
                # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8910555
                psd1 = self.psd(amp,0,4)
                psd2 = self.psd(amp,4,8)
                psd3 = self.psd(amp,8,12)
                psd4 = self.psd(amp,12,30)
                psd5 = self.psd(amp,30,50)
                psd6 = self.psd(amp,50,70)
                psd7 = self.psd(amp,70,100)
                
                psds = torch.stack((psd1, psd2, psd3, psd4, psd5, psd6, psd7))
                psd_sample.append(psds)

            psds_batch.append(torch.stack(psd_sample))

        return torch.stack(psds_batch)