
# Brandon Amos, J. Zico Kolter
# A PyTorch Implementation of DenseNet
# https://github.com/bamos/densenet.pytorch.

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, is_psd):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        
        if is_psd:
            self.net = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(),
            nn.Conv2d(nChannels, interChannels, kernel_size = 1, bias=False),
            nn.BatchNorm2d(interChannels),
            nn.ReLU(),
            nn.Conv2d(interChannels, growthRate, kernel_size=(1,25), 
                            padding=(0,12), bias=False)
        )

        else:
            self.net = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(),
            nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False),
            nn.BatchNorm2d(interChannels),
            nn.ReLU(),
            nn.Conv2d(interChannels, growthRate, kernel_size=(1,25), 
                               padding=(0,12), bias=False)
        )

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.net(x)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, is_psd):
        super(SingleLayer, self).__init__()

        if is_psd:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, growthRate, kernel_size=(1,25), 
                               padding=(0,12), bias=False)
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, growthRate, kernel_size=(1,25), 
                               padding=(0,12), bias=False)
            )

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        out = self.net(x)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, is_psd):
        super(Transition, self).__init__()
        
        self.is_psd = is_psd
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)        
        

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, (1,2))

        return out


class DENSENET_V3(nn.Module):
    def __init__(self, args, device):
        super(DENSENET_V3, self).__init__()
        self.args = args

        self.enc_model = self.args.enc_model
        self.growthRate = 12
        self.depth = 100
        self.reduction = 0.5
        self.nClasses = self.args.output_dim
        self.bottleneck = True
        self.num_data_channel = self.args.num_channel
        self.feature_extractor = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])

        nChannels = 2*self.growthRate

        self.is_psd = False 
        if self.enc_model == 'raw':
            self.features = False
            self.conv1 = nn.Conv2d(1, nChannels, kernel_size=(1,51), stride=(1,10), padding=(0,25), bias=False)
        else:
            self.features = True
            if self.enc_model == 'psd1' or self.enc_model =='psd2':
                self.is_psd = True
                self.conv1 = nn.Conv2d(self.num_data_channel, nChannels, kernel_size=(1,11), stride=(1,2), padding=(0,5),
                               bias=False)
            else:
                self.is_psd = False
                self.conv1 = nn.Conv2d(1, nChannels, kernel_size=(7,21), stride=(7,5), padding=(0,10),
                               bias=False)

        nDenseBlocks = (self.depth-4) // 3
        if self.bottleneck:
            nDenseBlocks //= 2


        self.dense1 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks*self.growthRate
        nOutChannels = int(math.floor(nChannels*self.reduction))
        self.trans1 = Transition(nChannels, nOutChannels, self.is_psd)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks*self.growthRate
        nOutChannels = int(math.floor(nChannels*self.reduction))
        self.trans2 = Transition(nChannels, nOutChannels, self.is_psd)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks*self.growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        # self.fc = nn.Linear(nChannels, self.nClasses)
        self.fc = nn.Linear(nChannels, self.nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, is_psd):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, is_psd))
            else:
                layers.append(SingleLayer(nChannels, growthRate, is_psd))
            nChannels += growthRate
        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        x = x.permute(0,2,1)
        if self.features:
            x = self.feature_extractor[self.enc_model](x)
            if self.args.enc_model == "sincnet":
                x = x.reshape(x.shape[0], 1, self.args.num_channel*self.args.sincnet_bandnum, x.shape[-1])
        else:
            x = torch.unsqueeze(x, dim=1)
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        out = nn.AdaptiveAvgPool2d((1,1))(F.relu(self.bn1(out)))
        out = torch.squeeze(out).view(out.shape[0], -1)
        out = self.fc(out)
        
        return out, 0
    
    def init_state(self, device):
        return 0