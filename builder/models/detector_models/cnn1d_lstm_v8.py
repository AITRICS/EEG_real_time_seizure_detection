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
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE
from builder.models.feature_extractor.lfcc_feature import LFCC_FEATURE

class CNN1D_LSTM_V8(nn.Module):
        def __init__(self, args, device):
                super(CNN1D_LSTM_V8, self).__init__()      
                self.args = args

                self.num_layers = args.num_layers
                self.hidden_dim = 256
                self.dropout = args.dropout
                self.num_data_channel = args.num_channel
                self.sincnet_bandnum = args.sincnet_bandnum
                
                self.feature_extractor = args.enc_model

                if self.feature_extractor == "raw" or self.feature_extractor == "downsampled":
                        pass
                else:
                        self.feat_models = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['LFCC', LFCC_FEATURE()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])
                        self.feat_model = self.feat_models[self.feature_extractor]

                if args.enc_model == "psd1" or args.enc_model == "psd2":
                        self.feature_num = 7
                elif args.enc_model == "LFCC":
                        self.feature_num = 8
                elif args.enc_model == "sincnet":
                        self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]
                elif args.enc_model == "stft1":
                        self.feature_num = 50
                elif args.enc_model == "stft2":
                        self.feature_num = 100
                elif args.enc_model == "raw" or args.enc_model == "downsampled":
                        self.feature_num = 1
                
                self.conv1dconcat_len = self.feature_num * self.num_data_channel
                
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
                self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device),
                        torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))

                def conv1d_bn(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm1d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                )
                def conv1d_bn_nodr(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm1d(oup),
                                self.activations[activation],
                )

                if args.enc_model == "raw":
                        self.features = nn.Sequential(
                        conv1d_bn(self.conv1dconcat_len,  64, 51, 4, 25),
                        nn.MaxPool1d(kernel_size=4, stride=4),
                        conv1d_bn(64, 128, 21, 2, 10),
                        conv1d_bn(128, 256, 9, 2, 4),
                        )
                elif args.enc_model == "sincnet":
                        self.features = nn.Sequential(
                        conv1d_bn(self.conv1dconcat_len,  64, 21, 2, 10),
                        conv1d_bn(64, 128, 21, 2, 10),
                        nn.MaxPool1d(kernel_size=4, stride=4),
                        conv1d_bn(128, 256, 9, 2, 4),
                        )
                elif args.enc_model == "psd1" or args.enc_model == "psd2":
                        self.features = nn.Sequential(
                                conv1d_bn(self.conv1dconcat_len,  64, 21, 2, 10), 
                                conv1d_bn(64, 128, 21, 2, 10),
                                nn.MaxPool1d(kernel_size=2, stride=2),
                                conv1d_bn(128, 256, 9, 1, 4),
                        )
                elif args.enc_model == "LFCC":
                        self.features = nn.Sequential(
                                conv1d_bn(self.conv1dconcat_len,  64, 21, 2, 10), 
                                conv1d_bn(64, 128, 21, 2, 10),
                                nn.MaxPool1d(kernel_size=2, stride=2),
                                conv1d_bn(128, 256, 9, 1, 4),
                        )
                elif args.enc_model == "downsampled":
                        self.conv1d_200hz = conv1d_bn_nodr(self.conv1dconcat_len,  32, 51, 4, 25)
                        self.conv1d_100hz = conv1d_bn_nodr(self.conv1dconcat_len,  16, 51, 2, 25)
                        self.conv1d_50hz = conv1d_bn_nodr(self.conv1dconcat_len,  16, 51, 1, 25)

                        self.features = nn.Sequential(
                                nn.MaxPool1d(kernel_size=4, stride=4),
                                conv1d_bn(64, 128, 21, 2, 10),
                                conv1d_bn(128, 256, 9, 1, 4),
                        )
                else:
                        self.features = nn.Sequential(
                                conv1d_bn(self.conv1dconcat_len,  64, 21, 2, 10), 
                                conv1d_bn(64, 128, 21, 2, 10),
                                nn.MaxPool1d(kernel_size=2, stride=2),
                                conv1d_bn(128, 256, 9, 1, 4),
                        )

          
                self.agvpool = nn.AdaptiveAvgPool1d(1)

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
                
        def forward(self, x):
                x = x.permute(0, 2, 1)
                if self.feature_extractor == "downsampled":
                        x_200 = self.conv1d_200hz(x)
                        x_100 = self.conv1d_100hz(x[:,:,::2])
                        x_50 = self.conv1d_50hz(x[:,:,::4])
                        x = torch.cat((x_200, x_100, x_50), dim=1)
                elif self.feature_extractor != "raw":
                        x = self.feat_model(x)
                        x = torch.reshape(x, (self.args.batch_size, self.conv1dconcat_len, x.size(3)))
                x = self.features(x)
                x = self.agvpool(x)
                x = x.permute(0, 2, 1)
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
                output, self.hidden = self.lstm(x, self.hidden)    
                output = output[:,-1,:]
                output = self.classifier(output)
                return output, self.hidden

        def init_state(self, device):
                self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
         