# This code references https://github.com/dansuh17/alexnet-pytorch
# first conv kernel size changed to 11->15, other conv kernels 3->5, rfield changed to 79
# the rest are unchanged from the original manuscript
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary


from builder.models.feature_extractor.psd_feature import PSD_FEATURE
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class ALEXNET_V2(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, args, device):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(ALEXNET_V2, self).__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.args = args
        self.num_classes = self.args.output_dim
        self.in_channels = self.args.num_channel
        self.batch_size = args.batch_size
        if self.args.window_size == 2:
            self.output_size = 5*11
        else:
            self.output_size = 1*5

        if self.args.enc_model == "sincnet" or self.args.enc_model == "psd":
            self.features = True
            self.feature_extractor = nn.ModuleDict({
                "psd" : PSD_FEATURE(),
                "sincnet" : SINCNET_FEATURE(args, num_eeg_channel=self.in_channels)
            })
            
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4)
            if self.args.window_size == 2:
                self.output_size = 5*5
            else:
                self.output_size = 5*2
            self.fc1 = nn.Linear(in_features=256 * self.output_size, out_features=4096)

        else:
            self.features = False
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,15), stride=(1,4))
            if self.args.window_size == 2:
                self.output_size = 5*11
            else:
                self.output_size = 1*5
            self.fc1 = nn.Linear(in_features=256*self.output_size, out_features=1024)

        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=2)
        self.net = nn.Sequential(
            # nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=(1,2), stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(128, 256, (1,5), padding=(0,2)),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(1,2), stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, (1,5), padding=(0,2)),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, (1,5), padding=(0,2)),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, (1,5), padding=(0,2)),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2),  # (b x 256 x 6 x 6)
        )
        self.dropout = nn.Dropout(p=0.5, inplace=True)

        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=256*5*2, out_features=4096),
            nn.ReLU(),
            nn.Dropout(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes),
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

        # print("\ninitial x shape,", x.shape)
        x = x.permute(0,2,1)

        if self.features:
            x = self.feature_extractor[self.args.enc_model](x)
        else:
            x = torch.unsqueeze(x, dim=1)

        x = self.conv1(x)

        x = self.net(x)

        # _, a,b,c = x.shape
        x = x.reshape(self.batch_size, -1)

        x = self.fc1(x)
       
        return self.classifier(x), 0

    def init_state(self, device):
        return 0