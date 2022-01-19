import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from builder.models.feature_extractor.psd_feature import PSD_FEATURE
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE


class MOBILENET1_2D_LSTM(nn.Module):
        def __init__(self, args, device):
                super(MOBILENET1_2D_LSTM, self).__init__()      
                self.args = args

                self.module_dict = nn.ModuleDict()

                self.num_layers = args.num_layers
                self.hidden_dim = args.hidden_dim
                self.num_data_channel = args.num_channel

                self.sincnet_bandnum = args.sincnet_bandnum
                
                feature_extractor = args.enc_model
                self.feat_models = nn.ModuleDict([
                        ['psd', PSD_FEATURE()],
                        ['sincnet', SINCNET_FEATURE(args=args,
                                                num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                ]])
                self.feat_model = self.feat_models[feature_extractor]

                if args.enc_model == "psd":
                        self.feature_num = 7
                elif args.enc_model == "sincnet":
                        self.feature_num = args.sincnet_bandnum

                # Create a new variable for the hidden state, necessary to calculate the gradients
                self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))

                def conv_bn(inp, oup, stride):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                                nn.BatchNorm2d(oup),
                                nn.ReLU(inplace=True)
                        )

                def conv_dw(inp, oup, stride):
                        return nn.Sequential(
                                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                                nn.BatchNorm2d(inp),
                                nn.ReLU(inplace=True),                
                                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(oup),
                                nn.ReLU(inplace=True),
                        )
                
                self.model = nn.Sequential(
                        conv_bn(self.num_data_channel,  32, 2), 
                        conv_dw( 32,  64, 2),
                        conv_dw( 64, 128, 2),
                        conv_dw(128, 128, 1),
                        conv_dw(128, 256, 2),
                        conv_dw(256, 256, 1),
                        conv_dw(256, 512, 2),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 1024, 2),
                        conv_dw(1024, 1024, 1),
                        # nn.AdaptiveAvgPool2d((1,1)),
                        )
        
                self.agvpool = nn.AdaptiveAvgPool2d((1,1))

                self.lstm = nn.LSTM(
                        input_size=1024,
                        hidden_size=self.hidden_dim,
                        num_layers=args.num_layers,
                        batch_first=True,
                        dropout=args.dropout) 
        
                self.linear_mean1 = nn.Linear(in_features=args.hidden_dim, out_features= 512, bias=True)
                self.bn1 = nn.BatchNorm1d(512)
                self.linear_mean2 = nn.Linear(in_features=512, out_features= args.output_dim, bias=True)
        
        def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.feat_model(x)
                x = self.model(x)
                x = self.agvpool(x)
                x = torch.squeeze(x, 2).permute(0, 2, 1)
                self.hidden = tuple(([Variable(var.data) for var in self.hidden]))

                output, self.hidden = self.lstm(x, self.hidden)    
                output = output[:,-1,:]
                output = self.linear_mean1(output)
                output = self.bn1(output)
                output = F.relu(output)
                logit = self.linear_mean2(output)
                # proba = nn.functional.softmax(output, dim=1)

                return logit, self.hidden
                
        def init_state(self, device):
                self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
         
        