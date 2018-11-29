#match_nn
#T1 need to fixed
import torch
import torch.nn as nn
import torchvision

class VideoNN(nn.Module):
    """
    customize network
    input T1 x D1

    output T1 x C
    """

    def __init__(self,T1,D_dim = 1000):
        """
        A convolution layer and many fc layers
        T1 >=16
        D_dim = 1000
        maxpool=[10,5,5,4]
        """
        super(VideoNN,self).__init__()
        
        self.p = [1,5,20,50] #i-p  i+p
        self.maxpool_k = [10,5,5,4]#as for D_dim = 1000
        self.T1 = T1
        self.convlayer = nn.Sequential(
            #conv
            #1 x T1 x D1 to 64 x T1 x D2
            nn.Conv2d(1,64,(1,2*self.p[0]+1),padding = (0,self.p[0])), 
            nn.ReLU(),
            nn.Conv2d(64,64,(1,2*self.p[0]+1),padding = (0,self.p[0])), 
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[0],1),stride=(self.maxpool_k[0],1)),
            #64 x T1 x D2 to 128 x T1 x D3
            nn.Conv2d(64,128,(1,2*self.p[1]+1),padding = (0,self.p[1])),
            nn.ReLU(),
            nn.Conv2d(128,128,(1,2*self.p[1]+1),padding = (0,self.p[1])), 
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[1],1),stride=(self.maxpool_k[1],1)),
            #128 x T1 x D3 to 256 x T1 x D4
            nn.Conv2d(128,256,(1,2*self.p[2]+1),padding = (0,self.p[2])),
            nn.ReLU(),
            nn.Conv2d(256,256,(1,2*self.p[2]+1),padding = (0,self.p[2])),
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[2],1),stride=(self.maxpool_k[2],1)),
            #128 x T1 x D4 to 512 x T1 x D5
            nn.Conv2d(256,512,(1,2*self.p[3]+1),padding = (0,self.p[3])),
            nn.ReLU(),
            nn.Conv2d(512,512,(1,2*self.p[3]+1),padding = (0,self.p[3])),
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[3],1),stride=(self.maxpool_k[3],1)),
            #512 x T1 x D5 to 1 x T1 x 1
            nn.Conv2d(512,1,1),
            nn.ReLU()
        )
        self.fclayer = nn.Sequential(
            nn.Linear(self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,self.T1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.convlayer(x)
        output = self.fclayer(x)
        return output

'''
class AudioNN(nn.Module):
    """
    customize network
    input T1 x D1

    output T1 x C
    """

    def __init__(self,D_dim):
        """
        A convolution layer and many fc layers

        D_dim
        maxpool=[10,5,5,4]  need to modify !!!!!!!!!!!!!!!!!!
        """
        super(AudioNN,self).__init__()
        
        self.p = [1,2,4,8]#i-p  i+p
        self.maxpool_k = [10,5,5,4]#as for D_dim = 1000
        self.T1 = opt.time
        self.convlayer = nn.Sequential(
            #conv
            #1 x T1 x D1 to 64 x T1 x D2
            nn.Conv2d(1,64,(2*self.p[0],1),padding = (self.p[0],0)), 
            nn.ReLU(),
            nn.Conv2d(64,64,(2*self.p[0],1),padding = (self.p[0],0)), 
            nn.ReLU(),
            nn.MaxPool2d((1,self.maxpool_k[0]),stride=(1,self.maxpool_k[0])),
            #64 x T1 x D2 to 128 x T1 x D3
            nn.Conv2d(64,128,(2*self.p[1],1),padding = (self.p[1],0)),
            nn.ReLU(),
            nn.Conv2d(128,128,(2*self.p[1],1),padding = (self.p[1],0)), 
            nn.ReLU(),
            nn.MaxPool2d((1,self.maxpool_k[1]),stride=(1,self.maxpool_k[1])),
            #128 x T1 x D3 to 256 x T1 x D4
            nn.Conv2d(128,256,(2*self.p[2],1),padding = (self.p[2],0)),
            nn.ReLU(),
            nn.Conv2d(256,256,(2*self.p[2],1),padding = (self.p[2],0)),
            nn.ReLU(),
            nn.MaxPool2d((1,self.maxpool_k[2]),stride=(1,self.maxpool_k[2])),
            #128 x T1 x D4 to 512 x T1 x D5
            nn.Conv2d(256,512,(2*self.p[3],1),padding = (self.p[3],0)),
            nn.ReLU(),
            nn.Conv2d(512,512,(2*self.p[3],1),padding = (self.p[3],0)),
            nn.ReLU(),
            nn.MaxPool2d((1,self.maxpool_k[3]),stride=(1,self.maxpool_k[3])),
            #512 x T1 x D5 to 1 x T1 x 1
            nn.Conv2d(512,1,1),
            nn.ReLU()
        )
        self.fclayer = nn.Sequential(
            nn.Linear(self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,self.T1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.convlayer(x)
        output = self.fclayer(x)
        return output
'''
class FeatureVideoNN(nn.Module):
    """
    customize network
    input T1 x D1(3)

    output T1 x C
    """

    def __init__(self,T1,D_dim = 3):
        """
        A convolution layer and many fc layers

        D_dim = 3 or small
        """
        super(FeatureVideoNN,self).__init__()
        
        self.p = [1,2,4,8] #i-p  i+p
        self.T1 = T1
        self.maxpool_k = [10,5,5,4]
        self.convlayer = nn.Sequential(
            #conv
            #1 x T1 x D1 to 64 x T1 x D2
            nn.Conv2d(1,64,(1,2*self.p[0]),padding = (0,self.p[0])), 
            nn.ReLU(),
            nn.Conv2d(64,64,(1,2*self.p[0]),padding = (0,self.p[0])), 
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[0],1),stride=(self.maxpool_k[0],1)),
            #64 x T1 x D2 to 128 x T1 x D3
            nn.Conv2d(64,128,(1,2*self.p[1]),padding = (0,self.p[1])),
            nn.ReLU(),
            nn.Conv2d(128,128,(1,2*self.p[1]),padding = (0,self.p[1])), 
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[1],1),stride=(self.maxpool_k[1],1)),
            #128 x T1 x D3 to 256 x T1 x D4
            nn.Conv2d(128,256,(1,2*self.p[2]),padding = (0,self.p[2])),
            nn.ReLU(),
            nn.Conv2d(256,256,(1,2*self.p[2]),padding = (0,self.p[2])),
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[2],1),stride=(self.maxpool_k[2],1)),
            #128 x T1 x D4 to 512 x T1 x D5
            nn.Conv2d(256,512,(1,2*self.p[3]),padding = (0,self.p[3])),
            nn.ReLU(),
            nn.Conv2d(512,512,(1,2*self.p[3]),padding = (0,self.p[3])),
            nn.ReLU(),
            nn.MaxPool2d((self.maxpool_k[3],1),stride=(self.maxpool_k[3],1)),
            #512 x T1 x D5 to 1 x T1 x 1
            nn.Conv2d(512,1,1),
            nn.ReLU()
        )
        self.fclayer = nn.Sequential(
            nn.Linear(self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,4*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4*self.T1,2*self.T1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*self.T1,self.T1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.convlayer(x)
        output = self.fclayer(x)
        return output



