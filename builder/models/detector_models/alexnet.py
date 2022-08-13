# Code reference: https://github.com/dansuh17/alexnet-pytorch


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import torch.optim as optim

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class ALEXNET_V4(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, args, device):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(ALEXNET_V4, self).__init__()
    
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
                nn.ReLU(),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),  # applies normalization across channels
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),  
                nn.Conv2d(96, 128, (1,7), padding=(0,3)),  
                nn.ReLU(),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,1)),  
                nn.Conv2d(128, 256, (1,7), padding=(0,3)), 
                nn.ReLU(),
                nn.Conv2d(256, 128, (1,7), padding=(0,3)),  
                nn.ReLU(),
                nn.Conv2d(128, 64, (1,7), padding=(0,3)),  
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),  
            )
            self.output_size = 7*1
        else:
            self.net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),  
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),  
                nn.Conv2d(64, 128, (1,15), padding=(0,7)),  
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),  
                nn.Conv2d(128, 256, (1,15), padding=(0,7)),  
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, (1,15), padding=(0,7)),  
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, (1,15), padding=(0,7)), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)), 
            )

            if self.enc_model == 'raw':
                self.features = False
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,51), stride=(1,4)) 
                self.output_size = 19*5
            else:
                if self.enc_model == 'sincnet':
                    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,21), stride=(7,2))
                    self.output_size = 20*5
                elif self.enc_model == 'stft1':
                    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2)
                    self.output_size = 2*2
                elif self.enc_model == 'stft2':
                    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2)
                    self.output_size = 5*2
                else:
                    print('Unsupported feature extractor chosen...')
        in_feature = 64 * self.output_size
        self.fc1 = nn.Linear(in_features=64 * self.output_size, out_features=in_feature//2) 
       
        self.classifier = nn.Sequential(
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

        x = x.permute(0,2,1)

        if self.args.enc_model != "raw":
            x = self.feature_extractor[self.args.enc_model](x)
            if self.args.enc_model == "sincnet":
                x = x.reshape(x.shape[0], 1, self.args.num_channel*self.args.sincnet_bandnum, x.shape[-1])
        else:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.classifier(x)

        return x, 0

    def init_state(self, device):
        return 0
