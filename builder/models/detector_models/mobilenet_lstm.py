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

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# Squeeze and excite layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def convbn(inp, oup, kernel, stride, pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, (1, kernel), (1, stride), (0,pad), bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
           self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (1,kernel_size), (1,stride), (0,(kernel_size - 1) // 2), groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, (1,kernel_size), (1,stride), (0,(kernel_size - 1) // 2), groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:

            return x + self.conv(x)
        else:
            x = self.conv(x)
            return x

class CNN2D_LSTM_V8_10(nn.Module):
        def __init__(self, args, device):
                super(CNN2D_LSTM_V8_10, self).__init__()      
                self.args = args

                self.num_layers = args.num_layers
                self.hidden_dim = 256
                self.dropout = args.dropout
                self.num_data_channel = args.num_channel
                self.sincnet_bandnum = args.sincnet_bandnum
                
                self.feature_extractor = args.enc_model

                if self.feature_extractor == "raw":
                        pass
                else:
                        self.feat_models = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])
                        self.feat_model = self.feat_models[self.feature_extractor]

                if args.enc_model == "psd1" or args.enc_model == "psd2":
                        self.feature_num = 7
                elif args.enc_model == "sincnet":
                        self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]
                elif args.enc_model == "stft1":
                        self.feature_num = 50
                elif args.enc_model == "stft2":
                        self.feature_num = 100
                elif args.enc_model == "raw":
                        self.feature_num = 1
                        self.num_data_channel = 1
                self.in_planes = 64
                
                activation = 'relu'
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()],
                        ['relu', nn.ReLU(inplace=True)],
                        ['tanh', nn.Tanh()],
                        ['sigmoid', nn.Sigmoid()],
                        ['leaky_relu', nn.LeakyReLU(0.2)],
                        ['elu', nn.ELU()]
                ])

                # Create a new variable for the hidden state, necessary to calculate the gradients
                self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))
                
                block = InvertedResidual

                self.conv1 = convbn(self.num_data_channel, 32, 51, 4, 25)
                self.feature = nn.Sequential(block(32, 32, 16, 9, 1, 0, 0),
                                block(16, 64, 32, 9, 2, 1, 0),
                                block(32, 80, 64, 9, 2, 0, 1),
                                block(64, 160, 128, 9, 2, 1, 1),
                                block(128, 384, 256, 9, 2, 1, 1),
                                )

                self.agvpool = nn.AdaptiveAvgPool2d((1,1))

                self.lstm = nn.LSTM(
                        input_size=256,
                        hidden_size=self.hidden_dim,
                        num_layers=args.num_layers,
                        batch_first=True,
                        dropout=args.dropout) 

                self.classifier = nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features= 64, bias=True),
                        nn.BatchNorm1d(64),
                        self.activations[activation],
                        nn.Linear(in_features=64, out_features= args.output_dim, bias=True),
                )

        def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride1 in strides:
                        layers.append(block(self.in_planes, planes, stride1))
                        self.in_planes = planes
                return nn.Sequential(*layers)
        
        def forward(self, x):
                x = x.permute(0, 2, 1)
                if self.feature_extractor != "raw":
                        x = self.feat_model(x)
                        x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
                else:
                        x = x.unsqueeze(1)
                        x = self.conv1(x)
                x = self.feature(x)
                x = self.agvpool(x)
                x = torch.squeeze(x, 2)
                x = x.permute(0, 2, 1)
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
                output, self.hidden = self.lstm(x, self.hidden)    
                output = output[:,-1,:]
                output = self.classifier(output)
                return output, self.hidden
                
        def init_state(self, device):
                self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
         
                


