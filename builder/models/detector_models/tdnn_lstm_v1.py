#pytorch tdnn model from https://github.com/cvqluu/TDNN/blob/master/tdnn.py
#tdnn+lstm example from https://github.com/kaldi-asr/kaldi/blob/master/egs/aspire/s5/local/chain/tuning/run_tdnn_lstm_1a.sh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class TDNN(nn.Module):    
    def __init__(
                    self, 
                    context_size,
                    dilation,
                    input_dim,
                    output_dim=512,
                    # context_size=5,
                    stride=1,
                    # dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)\

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        # x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        x = self.nonlinearity(x)

        return x



class TDNN_LSTM_V1(nn.Module):
    def __init__(self, args, device):
        super(TDNN_LSTM_V1, self).__init__()
        self.args = args

        self.num_layers = self.args.num_layers
        self.hidden_dim = self.args.hidden_dim
        self.num_data_channel = self.args.num_channel
        self.tdnn_input_dim = self.args.sincnet_bandnum
        self.sincnet_bandnum = args.sincnet_bandnum
                
        self.feature_extractor = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                                                        ]])
        self.feat_model = self.feature_extractor[args.enc_model]

        if args.enc_model == "psd":
            self.T = 200
            self.feature_num = 7
        elif args.enc_model == "sincnet":
            self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]
            self.T = 280
        self.conv1dconcat_len = self.feature_num * self.num_data_channel
        self.D = self.feature_num
        activation = 'relu'
        
        self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()],
                        ['relu', nn.ReLU(inplace=True)]
        ])
        
        

        self.frame1 = TDNN(input_dim=self.tdnn_input_dim, output_dim=256, context_size=2, dilation=1)
        self.frame2 = TDNN(input_dim=256, output_dim=256, context_size=4, dilation=2)
        self.frame3 = TDNN(input_dim=256, output_dim=256, context_size=1, dilation=1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.lstm1 = nn.LSTM(
                        input_size=256, 
                        hidden_size=128,
                        num_layers=2,
                        batch_first=True,
                        dropout=0.1,
                        proj_size=0) 
       
        self.classifier = nn.Sequential(
                        nn.Linear(in_features=128, out_features=32, bias=True),
                        nn.Sigmoid(),
                        nn.Dropout(0.2),
                        nn.Linear(in_features=32, out_features= 32, bias=True),
                        nn.Sigmoid(),
                        nn.Linear(in_features=32, out_features= args.output_dim, bias=True)
                )
        
        self.hidden = ((torch.zeros(2, self.args.batch_size, 128).to(device),
                torch.zeros(2, self.args.batch_size, 128).to(device)))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feat_model(x) 
  

        # Start first stack
        x_temp = x[:,0,:,:].permute(0,2,1)
        out = self.frame1(x_temp)
        out = self.frame2(out)
        out = self.frame3(out)
        out = self.avgpool(out.permute(0,2,1)) 
        # Compute TDNN layers for each channel
        # for channel in range(1,self.num_data_channel):
        for channel in range(1, self.num_data_channel):
            x_tdnn = x[:,channel,:,:].permute(0,2,1)
            x_tdnn = self.frame1(x_tdnn)
            x_tdnn = self.frame2(x_tdnn)
            x_tdnn = self.frame3(x_tdnn) 
            x_tdnn = self.avgpool(x_tdnn.permute(0,2,1))
            out = torch.cat((out, x_tdnn), dim=2)
        out = out.permute(0,2,1)
        hidden1 = tuple(([Variable(var.data) for var in self.hidden]))
        out, hidden1 = self.lstm1(out, hidden1)
        out = out[:,-1,:]
        logit = self.classifier(out)
        self.hidden = hidden1
        return logit, self.hidden

    def init_state(self, device):
        self.hidden = ((torch.zeros(2, self.args.batch_size, 128).to(device),
                torch.zeros(2, self.args.batch_size, 128).to(device)))

         
