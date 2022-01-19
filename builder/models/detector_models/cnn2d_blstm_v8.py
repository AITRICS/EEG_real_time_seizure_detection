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
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class CNN2D_BLSTM_V8(nn.Module):
        def __init__(self, args, device):
                super(CNN2D_BLSTM_V8, self).__init__()      
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

                h0 = torch.zeros(2, args.batch_size, self.hidden_dim//2).to(device) # 2 for bidirection 
                c0 = torch.zeros(2, args.batch_size, self.hidden_dim//2).to(device)
                self.blstm_hidden = (h0, c0)        

                def conv2d_bn(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                )

                if args.enc_model == "raw":
                        self.features = nn.Sequential(
                                conv2d_bn(self.num_data_channel,  64, (1,51), (1,4), (0,25)), 
                                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                                conv2d_bn(64, 128, (1,21), (1,2), (0,10)),
                                conv2d_bn(128, 256, (1,9), (1,2), (0,4)),
                        )
                elif args.enc_model == "sincnet":
                        self.features = nn.Sequential(
                                conv2d_bn(1,  64, (7,21), (7,2), (0,10)), 
                                conv2d_bn(64, 128, (1,21), (1,2), (0,10)),
                                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                                conv2d_bn(128, 256, (1,9), (1,2), (0,4)),
                        )
                elif args.enc_model == "psd1" or args.enc_model == "psd2":
                        self.features = nn.Sequential(
                                conv2d_bn(1,  64, (7,21), (7,2), (0,10)), 
                                conv2d_bn(64, 128, (1,21), (1,2), (0,10)),
                                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
                                conv2d_bn(128, 256, (1,9), (1,1), (0,4)),
                        )

                else:
                        self.features = nn.Sequential(
                                conv2d_bn(1,  64, (100,3), (100,1), (0,1)), 
                                conv2d_bn(64, 128, (1,51), (1,2), (0,25)),
                                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                                conv2d_bn(128, 256, (1,9), (1,1), (0,4)),
                        )
                self.agvpool = nn.AdaptiveAvgPool2d((1,8))

                self.blstm = nn.LSTM(
                        input_size=256,
                        hidden_size=self.hidden_dim//2,
                        num_layers=1,
                        batch_first=True,
                        bidirectional =True) 

                self.lstm = nn.LSTM(
                        input_size=self.hidden_dim ,
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

        def forward(self, x):#, blstm_hidden, prev_state=None):
                x = x.permute(0, 2, 1)
                if self.feature_extractor != "raw":
                        x = self.feat_model(x)
                        x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
                else:
                        x = x.unsqueeze(1)
                x = self.features(x)
                x = self.agvpool(x)
                x = torch.squeeze(x, 2)
                x = x.permute(0, 2, 1)
                output, _ = self.blstm(x, self.blstm_hidden)    
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
                output, self.hidden = self.lstm(output, self.hidden)    
                output = output[:,-1,:]
                output = self.classifier(output)
                return output, self.hidden

        def init_state(self, device):
                self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), 
                                torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
                