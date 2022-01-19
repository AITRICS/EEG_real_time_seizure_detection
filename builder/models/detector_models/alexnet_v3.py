# This code references https://github.com/dansuh17/alexnet-pytorch
# tried to change all conv kernels to 5->7, but input size too small
# one possible way to do that is to reduce pooling layer kernels and strides...
# alexnet_v3 : all conv kernels 7, first conv layer 15. r field 111
# Pooling layer kernel 3->2, stride 2->1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class ALEXNET_V3(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, args, device):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(ALEXNET_V3, self).__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.args = args
        self.num_classes = self.args.output_dim
        self.in_channels = self.args.num_channel
        self.enc_model = self.args.enc_model
        self.features = True
        self.num_data_channel = self.args.num_channel

        self.feature_extractor = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])

        if self.enc_model == 'psd1' or self.enc_model =='psd2':
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(1,21), stride=(1,4))
            self.net = nn.Sequential(
                # nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # applies normalization across channels
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),  # originally k=3, s=2
                nn.Conv2d(96, 128, (1,7), padding=(0,3)),  # (b x 256 x 27 x 27)
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,1)),  # (b x 256 x 13 x 13)
                nn.Conv2d(128, 256, (1,7), padding=(0,3)),  # (b x 384 x 13 x 13)
                nn.ReLU(),
                nn.Conv2d(256, 128, (1,7), padding=(0,3)),  # (b x 384 x 13 x 13)
                nn.ReLU(),
                nn.Conv2d(128, 64, (1,7), padding=(0,3)),  # (b x 256 x 13 x 13)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),  # (b x 256 x 6 x 6)
            )
            self.output_size = 7*1
        else:
            self.net = nn.Sequential(
                # nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4),  
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
                nn.MaxPool2d(kernel_size=2, stride=2),  
                nn.Conv2d(96, 128, 5, padding=2),  
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=2, stride=2),  
                nn.Conv2d(128, 256, 5, padding=2),  
                nn.ReLU(),
                nn.Conv2d(256, 128, 5, padding=2),  
                nn.ReLU(),
                nn.Conv2d(128, 64, 5, padding=2), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 
            )

            if self.enc_model == 'raw':
                self.features = False
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(1,21), stride=(1,4)) # compress first
                self.output_size = 2*5
            else:
                # self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4)
                if self.enc_model == 'sincnet':
                    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(1,21), stride=(1,4))
                    self.output_size = 2*2
                elif self.enc_model == 'stft1':
                    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2)
                    self.output_size = 2*2
                elif self.enc_model == 'stft2':
                    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2)
                    self.output_size = 5*2
                else:
                    print('Unsupported feature extractor chosen...')
        # self.fc1 = nn.Linear(in_features=64 * self.output_size, out_features=512) 
        in_feature = 64 * self.output_size
        self.fc1 = nn.Linear(in_features=64 * self.output_size, out_features=in_feature//2) 

        # classifier is just a name for linear layers
        # self.classifier = nn.Sequential(
        #     # nn.Linear(in_features=256*5*2, out_features=4096),
        #     nn.ReLU(),
        #     nn.Dropout(inplace=True),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.num_classes),
        # )
        # adaptivel feature size
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=256*5*2, out_features=4096),
            nn.ReLU(),
            nn.Dropout(inplace=True),
            nn.Linear(in_feature//2, in_feature//2),
            nn.ReLU(),
            nn.Linear(in_feature//2, self.num_classes),
        )

        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[9].bias, 1)
        nn.init.constant_(self.net[11].bias, 1)


    def forward(self, x):
        # """
        # Pass the input through the net.
        # Args:
        #     x (Tensor): input tensor
        # Returns:
        #     output (Tensor): output tensor
        # """

        # print("\n0, ", x.shape)
        x = x.permute(0,2,1)

        if self.features:
            x = self.feature_extractor[self.args.enc_model](x)
        else:
            x = torch.unsqueeze(x, dim=1)
        # print("1, ", x.shape)
        x = self.conv1(x)
        # print("2, ", x.shape)
        x = self.net(x)
        # print("3, ", x.shape)
        x = x.view(x.shape[0], -1)
        # print("3.5, ", x.shape)
        x = self.fc1(x)
        # print("4, ", x.shape)
        x = self.classifier(x)
        # print("5, ", x.shape)
        # exit(1)
        return x, 0

    def init_state(self, device):
        return 0
