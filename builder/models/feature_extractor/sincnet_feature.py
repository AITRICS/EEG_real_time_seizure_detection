"""Sinc-based convolution
Mirco Ravanelli, Yoshua Bengio,
"Speaker Recognition from raw waveform with SincNet".
https://arxiv.org/abs/1808.00158
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from torch import Tensor

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

# class LayerNorm(nn.Module):
#     """ Wrapper class of torch.nn.LayerNorm """
#     def __init__(self, dim: int, eps: float = 1e-6) -> None:
#         super(LayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(dim))
#         self.beta = nn.Parameter(torch.zeros(dim))
#         self.eps = eps

#     def forward(self, z: Tensor) -> Tensor:
#         print("0: ", z.shape)
#         mean = z.mean(dim=-1, keepdim=True)
#         std = z.std(dim=-1, keepdim=True)
#         output = (z - mean) / (std + self.eps)
#         print("1: ", output.shape)
#         print("2: ", self.gamma.shape)
#         print("3: ", self.beta.shape)
#         output = self.gamma * output + self.beta

#         return output

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=200, in_channels=1,
                 stride=1, padding=0, normalize=None, slice_len=280, dilation=1, bias=False, groups=1, min_low_hz=0, min_band_hz=2):
        super(SincConv_fast, self).__init__()

        self.normalize = normalize
        if self.normalize == "layernorm":
            # self.ln0=nn.LayerNorm([1,slice_len])
            self.ln0=nn.LayerNorm(1)
            # self.ln0=LayerNorm(1)
        elif self.normalize == "batchnorm":
            self.bn0=nn.BatchNorm1d([1],momentum=0.05)

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes



    def forward(self, waveforms):
        """
        Parameters
        ----------
        eeg waveforms : `torch.Tensor` (batch_size, 1, n_samples)

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        waveforms = waveforms.unsqueeze(1)
        if self.normalize == "layernorm":
            waveforms = self.ln0(waveforms)
        elif self.normalize == "batchnorm":
            waveforms = self.bn0(waveforms)

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 

        band_pass_center = 2*band.view(-1,1)

        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
   
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(3,1,1)
        # plt.plot(band_pass[0].detach().cpu().numpy())
        # plt.subplot(3,1,2)
        # plt.plot(band_pass[5].detach().cpu().numpy())
        # plt.subplot(3,1,3)
        # plt.plot(band_pass[10].detach().cpu().numpy())
        # plt.show()
        # exit(1)

        band_pass = band_pass / (2*band[:,None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 

class sincnet_conv_layers(nn.Module):
    def __init__(self, args):
        super(sincnet_conv_layers, self).__init__()
        self.cnn_layer_num = args.sincnet_layer_num
        self.conv  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.maxpool = nn.ModuleList([])

        slice_len = args.window_size_sig + (args.sincnet_kernel_size -1)
        cnn_channel_sizes = args.cnn_channel_sizes
        cnn_kernel_sizes = [args.sincnet_kernel_size, 5, 5]
        self.stride_list = [args.sincnet_stride, 2, 2]
        current_input = args.window_size_sig + (args.sincnet_kernel_size -1)
        normalize = args.sincnet_input_normalize
        
        for i in range(self.cnn_layer_num):
            if i==0:
                self.conv.append(SincConv_fast(out_channels=cnn_channel_sizes[i], 
                                            kernel_size=cnn_kernel_sizes[i], 
                                            stride=self.stride_list[i], 
                                            padding=0, 
                                            normalize=normalize,
                                            slice_len=slice_len))
            else:
                self.conv.append(nn.Conv1d(in_channels=cnn_channel_sizes[i-1], 
                                            out_channels=cnn_channel_sizes[i], 
                                            kernel_size=cnn_kernel_sizes[i], 
                                            stride=self.stride_list[i]))
            
            # self.maxpool.append(nn.MaxPool1d(self.max_pool_list[i]))
            layernorm_output = int((current_input-cnn_kernel_sizes[i]+1)/self.stride_list[i])
            # self.ln.append(nn.LayerNorm([cnn_channel_sizes[i], layernorm_output]))
            self.ln.append(nn.LayerNorm(cnn_channel_sizes[i]))
            # self.ln.append(LayerNorm(cnn_channel_sizes[i]))
            self.act.append(nn.LeakyReLU(0.2))

            current_input = layernorm_output

    def forward(self, x):
        # print("x0: ", x.shape)
        for i in range(self.cnn_layer_num):
            if i==0:
                x = torch.abs(self.conv[i](x))
                x = x.permute(0,2,1)
                # x = self.act[i](self.ln[i](x.permute(0,2,1)).permute(0,2,1))
                x = self.act[i](self.ln[i](x))
            else:
                x = self.act[i](self.conv[i](x))
        return x


class SINCNET_FEATURE(nn.Module):
    def __init__(self, args, num_eeg_channel):
        super(SINCNET_FEATURE, self).__init__()
        
        # self.conv_list = nn.ModuleList()
        # for _ in range(num_eeg_channel):
        #     self.conv_list.append(sincnet_conv_layers(args))
        self.sinc_conv = sincnet_conv_layers(args)
        
    def forward(self, waveforms):
        output_list = []
        waveforms = waveforms.permute(1,0,2)
        # for idx, conv in enumerate(self.conv_list):
        #     output_list.append(conv(waveforms[idx]))
        for waveform in waveforms:
            output_list.append(self.sinc_conv(waveform))

        return torch.stack(output_list).permute(1,0,3,2)

