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
import platform
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from scipy import signal as sci_sig
import torchaudio

class LFCC_FEATURE(nn.Module):
    def __init__(self,
            sample_rate: int = 200,
            frame_length: int = 16,
            frame_shift: int = 8,
            feature_extract_by: str = 'kaldi'):
        super(LFCC_FEATURE, self).__init__()

        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 2

        assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
        self.transforms = torchaudio.transforms.LFCC(sample_rate = 200, 
                                    n_filter = 32,
                                    f_min = 0.0,    # 0 Hz ~
                                    f_max = 100,     # ~ 60Hz
                                    n_lfcc = 8,    # 8 coefficietns
                                    dct_type = 2, 
                                    norm = 'ortho', 
                                    log_lf = False,
                                    speckwargs = {"n_fft":400, "win_length":60, "hop_length":30}) # 300miliseconds --> 60 points for 200Hz, 150miliseconds --> 30 points for 200Hz

    def psd(self, amp, begin, end):
        return torch.mean(amp[begin*self.freq_resolution:end*self.freq_resolution], 0)
        
    def forward(self, batch):
        final_batch = []

        for signals in batch:
            transformed_sample = []
            for signal in signals:
                stft = self.transforms(signal)
                amp = (torch.log(torch.abs(stft) + 1e-10))
                    
                transformed_sample.append(amp)

            final_batch.append(torch.stack(transformed_sample))

        return torch.stack(final_batch)

