from builder.models.seizurenet import conv_bn_relu_pool
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class RESNET18_CONV1D_V1(nn.Module):
    def __init__(self, args, device):
        super(RESNET18_CONV1D_V1, self).__init__()
        self.args = args

        self.num_channel = self.args.num_channel
        self.dropout = self.args.dropout
        self.batch_size = self.args.batch_size
        self.out_channels = [40,40,80,160,320]
        self.output_dim = self.args.output_dim
        self.shortcut_method = "projection"

        self.conv1 = nn.Conv1d(self.num_channel, 40, 7, stride=2, padding=3)
        # self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.fc1 = nn.Linear(320, self.output_dim)
        self.model = nn.ModuleList()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        def _projection_shortcut(in_channel, out_channel):
            return nn.Conv1d(in_channel, out_channel, 1, stride=2)
                 
        def _conv_layer(inp, oup, kernel_size, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, kernel_size, stride, padding=1),
                nn.BatchNorm1d(oup)
            )
        
        # self.model = nn.Sequential(
        #     _conv_layer(20,40,3,2),
        #     _conv_layer(40,40,3,1),
        #     _conv_layer(40,80,3,2),
        #     _conv_layer(80,80,3,1),
        #     _conv_layer(80,120,3,2),
        #     _conv_layer(160,160,3,1),
        #     _conv_layer(160,320,3,2),
        #     _conv_layer(320,320,3,1)
        # )

        self.shortcut0 = _projection_shortcut(20,40)
        self.shortcut1 = _projection_shortcut(40,80)
        self.shortcut2 = _projection_shortcut(80,160)
        self.shortcut3 = _projection_shortcut(160,320)

        self.resnet_block_a_1 = _conv_layer(20,40,3,2)
        self.resnet_block_a_2 = _conv_layer(40,40,3,1)
        self.resnet_block_b_1 = _conv_layer(40,80,3,2)
        self.resnet_block_b_2 = _conv_layer(80,80,3,1)
        self.resnet_block_c_1 = _conv_layer(80,160,3,2)
        self.resnet_block_c_2 = _conv_layer(160,160,3,1)
        self.resnet_block_d_1 = _conv_layer(160,320,3,2)
        self.resnet_block_d_2 = _conv_layer(320,320,3,1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x_res1 = self.resnet_block_a_1(x)
        x = self.shortcut0(x)
        x = x+x_res1
        x_res2 = self.resnet_block_a_2(x)
        x = x+x_res2
        # print(f'6:_____{x.shape}')

        x_res3 = self.resnet_block_b_1(x)
        x = self.shortcut1(x)
        x = x+x_res3
        x_res4 = self.resnet_block_b_2(x)
        x = x+x_res4
        # print(f'1:_____{x.shape}')
        x_res5 = self.resnet_block_c_1(x)
        x = self.shortcut2(x)
        x = x+x_res5
        x_res6 = self.resnet_block_c_2(x)
        x = x+x_res6
        # print(f'5:_____{x.shape}')

        x_res7 = self.resnet_block_d_1(x)
        x = self.shortcut3(x)
        x = x+x_res7
        x_res8 = self.resnet_block_d_2(x)
        x = x+x_res8

        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        return x, 0

    def init_state(self, device):
        return 0