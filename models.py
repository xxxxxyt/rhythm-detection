#match_nn
#T need to fixed
import torch
import torch.nn as nn
import torchvision
from common import ChainCRF, Shift

class EndToEndCRF(nn.Module):
    def __init__(self, args):
        super(EndToEndCRF, self).__init__()

        self.T = args.segment_length
        self.D = args.dim_feature

        # vgg feature extraction
        # TODO
        # vgg_net.forward():
        # input.shape (n, T, H, W) or whatever you draw from video
        # output shape (n, T, D)
        self.vgg_net = None

        self.crf = CRF(args)    # CRF network, not CRF layer

    def loss(self, x, label):
        x = self.vgg_net(x)
        return self.crf.loss(x, label)

    def forward(self, x):
        x = self.vgg_net(x)
        return self.crf.forward(x)

class CRF(nn.Module):
    def __init__(self, args):
        super(CRF, self).__init__()

        self.D = args.dim_feature
        self.d_hidden = 256
        # self.shift = Shift(args)
        # self.linears = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.D, self.D), nn.ReLU(), nn.Dropout()
        # ) for i in range(2)])
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=self.d_hidden, 
            batch_first=True, bidirectional=True, num_layers=2, dropout=0.4)
        self.crf = ChainCRF(self.d_hidden * 2, 2)
        self.sigmoid = nn.Sigmoid()

    def _get_crf_input(self, x):
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
        label = label.clone().long()
        x = self._get_crf_input(x)
        return torch.sum(self.crf.loss(x, label[:,:,0]))

    def forward(self, x):
        """
        x.shape:        (n, T, D)
        return.shape:   (n, T, 1)
        """
        x = self._get_crf_input(x)
        return self.crf.decode(x).unsqueeze(dim=-1)
