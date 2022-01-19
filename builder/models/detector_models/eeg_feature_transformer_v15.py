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
from builder.models.src.transformer import *
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE
from torch import Tensor
from typing import Tuple, Optional, Any
import torch.nn.init as init
from builder.models.src.transformer.module import PositionalEncoding


class EEG_FEATURE_TRANSFORMER_V15(nn.Module):
    def __init__(self, args, device):
        super(EEG_FEATURE_TRANSFORMER_V15, self).__init__()      
        self.args = args

        self.num_layers = args.num_layers
        self.hidden_dim = 256
        self.dropout = args.dropout
        self.num_data_channel = args.num_channel
        self.sincnet_bandnum = args.sincnet_bandnum
        enc_model_dim = 128
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

        def conv2d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                    nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(oup),
                    self.activations[activation],
                    nn.Dropout(self.dropout))

        if args.enc_model == "raw":
            self.features = nn.Sequential(
                    conv2d_bn(self.num_data_channel, 64, (1,51), (1,4), (0,25)), 
                    nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                    conv2d_bn(64, 128, (1,21), (1,2), (0,10)),
                    conv2d_bn(128, 256, (1,9), (1,2), (0,4)),
                    nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            )
            transformer_d_input = 1536
        elif args.enc_model == "sincnet":
            self.features = nn.Sequential(
                conv2d_bn(1,  128, (7,21), (7,2), (0,10)), 
                conv2d_bn(128, 128, (1,21), (1,2), (0,10)),
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                conv2d_bn(128, 256, (1,9), (1,2), (0,4)),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            )
            transformer_d_input = 1536
        else:
            self.features = nn.Sequential(
                    conv2d_bn(self.feature_num, 128, (1,25), (1,3), (0,12)), 
                    nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                    conv2d_bn(128, 256, (1,13), 1, (0,6)),
                    conv2d_bn(256, 256, (1,7), 1, (0,3)),
            ) 
       
        self.transformer_encoder = TransformerEncoder(
                                                d_input= transformer_d_input, 
                                                n_layers= 4, 
                                                n_head= 4, 
                                                d_model= enc_model_dim,
                                                d_ff= enc_model_dim*4,
                                                dropout=0.1, 
                                                pe_maxlen=500,
                                                use_pe=False,
                                                block_mask=None)
        self.agvpool = nn.AdaptiveAvgPool1d(1)

        self.hidden = ((torch.zeros(1, args.batch_size, 20).to(device),
                        torch.zeros(1, args.batch_size, 20).to(device)))

        self.positional_encoding = PositionalEncoding(256, max_len=10)
        self.pe_x = self.positional_encoding(6).to(device)

        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=20,
            num_layers=1,
            batch_first=True) 

        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features= args.output_dim, bias=True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.feature_extractor != "raw":
            x = self.feat_model(x)
            x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
        else:
            x = x.unsqueeze(1)

        x = self.features(x).permute(0,2,3,1)

        x = x  + self.pe_x.unsqueeze(0)
        x = x.reshape(x.size(0), 20, -1)
        x = self.transformer_encoder(x)
        x = self.agvpool(x)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x.permute(0,2,1), self.hidden)    
        output = self.classifier(output.squeeze(1))
        return output, self.hidden

    def init_state(self, device):
        self.hidden = ((torch.zeros(1, self.args.batch_size, 20).to(device), torch.zeros(1, self.args.batch_size, 20).to(device)))