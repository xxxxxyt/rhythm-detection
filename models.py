import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from common import ChainCRF, Shift
from vgg_net import Vgg16Conv
from collections import OrderedDict

class EndToEnd(nn.Module):
    def __init__(self, args):
        super(EndToEnd, self).__init__()

        self.T = args.segment_length
        self.D = args.dim_feature

        self.vgg_net = Vgg16Conv(num_cls=args.dim_feature, init_weights=args.vgg_init)
        self.sequence_labeling = SequenceLabeling(args)

    def loss(self, x, label):
        x = self.vgg_net(x)
        x = self.sequence_labeling.loss(x, label)
        return x

    def forward(self, x):
        x = self.vgg_net(x)
        x = self.sequence_labeling.forward(x)
        return x

class SequenceLabeling(nn.Module):
    def __init__(self, args):
        super(SequenceLabeling, self).__init__()

        self.D = args.dim_feature
        self.d_hidden = 256
        self.use_crf = args.use_crf

        # self.shift = Shift(args)
        # self.linears = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.D, self.D), nn.ReLU(), nn.Dropout()
        # ) for i in range(2)])
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=self.d_hidden, 
            batch_first=True, bidirectional=True, num_layers=2, dropout=0.4)
        self.crf = ChainCRF(self.d_hidden * 2, 2)
        self.out = nn.Sequential(nn.Linear(self.d_hidden * 2, 1), nn.Sigmoid())

    def get_lstm_output(self, x):
        """
        x.shape:        (n, T, D)
        return.shape:   (n, T, d_hidden * 2)
        """
        # x = self.shift(x)
        # for linear in self.linears:
        #     x = linear(x)
        x, (h_n, c_n) = self.lstm(x) # (n, T, d_hidden * 2)
        return x

    def loss(self, x, label):
        """
        x.shape:        (n, T, D)
        label.shape:    (n, T, 1)
        return.shape:   (1,)
        """
        x = self.get_lstm_output(x)
        label = label.clone().long() if self.use_crf else \
                label.clone().float()
        loss = torch.sum(self.crf.loss(x, label[:,:,0])) if self.use_crf else \
               torch.nn.functional.binary_cross_entropy(self.out(x), label)
        return loss

    def forward(self, x):
        """
        x.shape:        (n, T, D)
        return.shape:   (n, T, 1)
        """
        x = self.get_lstm_output(x)
        x = self.crf.decode(x).unsqueeze(dim=-1) if self.use_crf else \
            self.out(x)
        return x
