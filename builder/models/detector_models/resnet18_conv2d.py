'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
'''
Code from https://github.com/kuangliu/pytorch-cifar
'''

# resnet18 adaptpooling 1x4

import torch
import torch.nn as nn
import torch.nn.functional as F

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, is_psd, stride=1):
        super(BasicBlock, self).__init__()

        if is_psd:
            self.net = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1,7), stride=(1,stride), padding=(0,3), bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(planes, planes, kernel_size=(1,7), stride=(1,1), padding=(0,3), bias=False),
                nn.BatchNorm2d(planes)
                
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(planes, planes, kernel_size=(1,9), stride=(1,1), padding=(0,4), bias=False),
                nn.BatchNorm2d(planes)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if is_psd:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=(1,stride), bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=(1,stride), bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = self.net(x)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RESNET18_CONV2D_V2_2(nn.Module):
    def __init__(self, args, device):
        super(RESNET18_CONV2D_V2_2, self).__init__()
        self.args = args

        num_classes = self.args.output_dim
        self.enc_model = self.args.enc_model
        # num_blocks = [4,4,4,4] # for rfield 109
        num_blocks = [2,2,2,2] # original rfield 49
        block = BasicBlock
        self.num_data_channel = self.args.num_channel
        self.in_planes = 64
        self.in_channels = self.args.num_channel
        self.feature_extractor = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])

        self.features = True
        self.is_psd = False

        if self.enc_model == 'psd1' or self.enc_model == 'psd2':
            self.is_psd = True
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(1,15), stride=(1,2), padding=(0,7), bias=False)
        else:
            if self.enc_model == 'raw':
                self.features = False
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,51), stride=(1,4), padding=(0,25), bias=False)
            elif self.enc_model == 'sincnet':
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,21), stride=(7,2), padding=(0,10), bias=False)
            else:
                self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], self.is_psd, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], self.is_psd, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], self.is_psd, stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], self.is_psd, stride=2)
        self.fc1 = nn.Linear(4*256*block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, is_psd, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, is_psd, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('0: ', x.shape)
        x = x.permute(0,2,1)
        if self.args.enc_model != "raw":
            x = self.feature_extractor[self.args.enc_model](x)
            if self.args.enc_model == "sincnet":
                x = x.reshape(x.shape[0], 1, self.args.num_channel*self.args.sincnet_bandnum, x.shape[-1])
        else:
            x = torch.unsqueeze(x, dim=1)
        # print('1: ', x.shape)
        x = self.conv1(x)
        # print('2: ', x.shape)
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print('3: ', out.shape)
        out = nn.AdaptiveAvgPool2d((1,4))(out)
        # print('4: ', out.shape)
        out = out.view(out.size(0), -1)
        # print('5: ', out.shape)
        out = self.fc1(out)
        # print('6: ', out.shape)
        # exit(1)
        return out, 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)



    def init_state(self ,device):
        return 0
# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])