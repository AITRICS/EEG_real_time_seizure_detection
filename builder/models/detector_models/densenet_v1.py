
# Brandon Amos, J. Zico Kolter
# A PyTorch Implementation of DenseNet
# https://github.com/bamos/densenet.pytorch.
# Accessed: [2021.10.18]

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

from builder.models.feature_extractor.psd_feature import PSD_FEATURE
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DENSENET_V1(nn.Module):
    def __init__(self, args, device):
        super(DENSENET_V1, self).__init__()
        self.args = args

        self.growthRate = 12
        self.depth = 100
        self.reduction = 0.5
        self.nClasses = self.args.output_dim
        self.bottleneck = True
        self.num_data_channel = self.args.num_channel

        if self.args.enc_model == "sincnet" or self.args.enc_model == "psd":
            self.features = True
            if args.enc_model == "psd":
                self.feature_extractor = PSD_FEATURE()
                self.feature_num = 7
            else:
                self.feature_extractor = SINCNET_FEATURE(args=args,
                                        num_eeg_channel=self.num_data_channel)
                self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]
        else:
            self.features = False

        nDenseBlocks = (self.depth-4) // 3
        if self.bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*self.growthRate
        
        if self.features:
            self.conv1 = nn.Conv2d(self.num_data_channel, nChannels, kernel_size=3, padding=1,
                               bias=False)
        else:
            self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck)
        nChannels += nDenseBlocks*self.growthRate
        nOutChannels = int(math.floor(nChannels*self.reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck)
        nChannels += nDenseBlocks*self.growthRate
        nOutChannels = int(math.floor(nChannels*self.reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck)
        nChannels += nDenseBlocks*self.growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, self.nClasses)
        print("nChannels: ", nChannels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0,2,1)
        if self.features:
            x = self.feature_extractor(x)
        else:
            x = torch.unsqueeze(x, dim=1)
        print("x shape raw: ", x.shape)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        print("shape out: ", out.shape)
        # out = F.avg_pool2d(F.relu(self.bn1(out)), 5)
        out = nn.AdaptiveAvgPool2d(1)(F.relu(self.bn1(out)))

        print("shape out after pool2d: ", out.shape)
        out = torch.squeeze(out)
        # out = F.log_softmax(self.fc(out))
        out = self.fc(out)
        return out, 0
    
    def init_state(self, device):
        return 0