# """
# @misc{chitlangia2021epileptic,
#   author = {Chitlangia, Sharad},
#   title = {Epileptic Seizure Detection using Deep Learning},
#   year = {2021},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/Sharad24/Epileptic-Seizure-Detection/}},
# }
# """

# import mne
import numpy as np
import pyedflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.gru1 = nn.GRU(22, 32, num_layers = 1, batch_first = True)
        self.affine1 = nn.Linear(15000, 1875)
        self.gru2 = nn.GRU(32, 32, num_layers = 2, batch_first = True)
        self.affine2 = nn.Linear(1875, 1)
        self.gru3 = nn.GRU(32, 2, num_layers = 1, batch_first = True)
        
    def forward(self, x):
        x = x.contiguous().view(64, 15000, 22)
        x, _ = self.gru1(x)
        x = x.contiguous().view(64, 32, 15000)
        x = F.elu(self.affine1(x))
        x = x.contiguous().view(64, 1875, 32)
        x, _ = self.gru2(x)
        x = x.contiguous().view(64, 32, 1875)
        x = F.elu(self.affine2(x))
        x = x.contiguous().view(64, 1, 32)
        x, _ = self.gru3(x)
        x = torch.squeeze(x, dim = 1)
        x = F.softmax(x)
        return x
    
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv1d(22, 32, 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv1d(32, 32, 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv1d(32, 32, 4, stride = 2, padding = 1)
        self.gru1 = nn.GRU(32, 32, num_layers = 3, batch_first = True)
        self.affine1 = nn.Linear(1875, 1)
        self.gru2 = nn.GRU(32, 2, batch_first = True)
    
    def forward(self, x):
        x = x.contiguous().view(64, 22, 15000)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.contiguous().view(64, 1875, 32)
        x, _ = self.gru1(x)
        x = x.contiguous().view(64, 32, 1875)
        x = F.elu(self.affine1(x))
        x = x.contiguous().view(64, 1, 32)
        x, _ = self.gru2(x)
        x = torch.squeeze(x, dim = 1)
        x = F.softmax(x)
        return x
    
    
class ICRNN(nn.Module):
    def __init__(self):
        super(ICRNN, self).__init__()
        self.inception1 = Inception(22)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)
        self.gru1 = nn.GRU(96, 32, num_layers = 3, batch_first = True)
        self.affine1 = nn.Linear(1875, 1)
        self.gru2 = nn.GRU(32, 2, batch_first = True)
        
    def forward(self, x):
        x = x.contiguous().view(64, 22, 15000)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = x.contiguous().view(64, 1875, 96)
        x, _ = self.gru1(x)
        x = x.contiguous().view(64, 32, 1875)
        x = self.affine1(x)
        x = x.contiguous().view(64, 1, 32)
        x, _ = self.gru2(x)
        x = torch.squeeze(x, dim = 1)
        x = F.softmax(x)
        return x
    
    
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv1d(in_channels, 32, 2, stride = 2)
        self.conv3 = nn.Conv1d(in_channels, 32, 8, stride = 2, padding = 3)
        
    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        cat = [x1, x2, x3]
        x = torch.cat(cat, dim=1)
        return x

class DCRNN(nn.Module): #Densely connected convolutional recurrent neural network
    def __init__(self):
        super(DCRNN, self).__init__()
        self.conv1 = nn.Conv1d(22, 32, 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv1d(32, 32, 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv1d(32, 32, 4, stride = 2, padding = 1)
        self.gru1 = nn.GRU(32, 32, num_layers = 3, batch_first = True)
        self.affine1 = nn.Linear(1875, 1)
        self.gru2 = nn.GRU(32, 32, batch_first = True)
        self.gru3 = nn.GRU(64, 32, batch_first = True)
        self.gru4 = nn.GRU(96, 32, batch_first = True)
    
    def forward(self, x):
        x = x.contiguous().view(64, 22, 15000)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.contiguous().view(64, 1875, 32)
        x, _ = self.gru1(x)
        x_res = x
        x, _ = self.gru2(x)
        x_res2 = x
        x_cat1 = torch.cat([x_res, x], dim = 2)
        x, _ = self.gru3(x_cat1)
        x = torch.cat([x_res, x_res2, x], dim = 2)
        x = x.contiguous().view(64, 96, 1875)
        x = F.elu(self.affine1(x))
        x = x.contiguous().view(64, 1, 96)
        x, _ = self.gru4(x)
        x = torch.squeeze(x, dim = 1)
        x = F.softmax(x)
        return x
    
    
class CHRONONET(nn.Module):
    def __init__(self, args, device):
        super(CHRONONET, self).__init__()
        self.args = args 
        self.batch_size = self.args.batch_size
        self.num_channel = self.args.num_channel
        self.slice_len = self.args.sample_rate * self.args.window_size
        self.out_channel = 96
        self.feature_len = [self.slice_len,100,500,25]

        self.hidden_gru1 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru2 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru3 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru4 = torch.zeros(1, self.args.batch_size, 2).to(device)

        self.inception1 = Inception(self.num_channel)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)
        self.gru1 = nn.GRU(96, 32, num_layers = 1, batch_first = True)
        self.affine1 = nn.Linear(100, 1)
        self.gru2 = nn.GRU(32, 32, batch_first = True)
        self.gru3 = nn.GRU(64, 32, batch_first = True)
        self.gru4 = nn.GRU(96, 2, batch_first = True)
        
    def forward(self, x):
        x = x.contiguous().view(self.batch_size, self.num_channel, self.slice_len)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)


        x = x.contiguous().view(self.batch_size, -1, self.out_channel) 
        self.hidden_gru1 = Variable(self.hidden_gru1.data)
        self.hidden_gru2 = Variable(self.hidden_gru2.data)
        self.hidden_gru3 = Variable(self.hidden_gru3.data)
        self.hidden_gru4 = Variable(self.hidden_gru4.data)

        x, self.hidden_gru1 = self.gru1(x, self.hidden_gru1) 
        x_res = x
        x, self.hidden_gru2 = self.gru2(x, self.hidden_gru2)
        x_res2 = x
        x_cat1 = torch.cat([x_res, x], dim = 2)

        x, self.hidden_gru3 = self.gru3(x_cat1, self.hidden_gru3)
        x = torch.cat([x_res, x_res2, x], dim = 2)
        x = x.contiguous().view(self.batch_size, self.out_channel, -1)
        x = F.elu(self.affine1(x))
        x = x.contiguous().view(self.batch_size, 1, self.out_channel)
        x, self.hidden_gru4 = self.gru4(x, self.hidden_gru4)
        x = torch.squeeze(x, dim = 1)

        return x, [self.hidden_gru1, self.hidden_gru2, self.hidden_gru3, self.hidden_gru4]
    
    def init_state(self, device):
        self.hidden_gru1 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru2 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru3 = torch.zeros(1, self.args.batch_size, 32).to(device)
        self.hidden_gru4 = torch.zeros(1, self.args.batch_size, 2).to(device)
