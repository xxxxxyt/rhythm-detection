
from nn import VideoNN
import argparse
import torch
from dataloader import *
from torch.utils.data import DataLoader
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

parser = argparse.ArgumentParser(description='PyTorch Training')
rootdir = os.getcwd()
parser.add_argument('--audiofeaturedir',dest='audio_feature_dir',type = str,
                    help='audiofeature-directory',default="")
parser.add_argument('--vggfeaturedir',dest='vgg_feature_dir',type = str,
                    help='vgg-directory',default="")
parser.add_argument('--audioformat',dest='audioformat',type = int,
                    help='audio_format 1,2',default=1)
parser.add_argument('--batchsize',dest='batchsize',type = int,
                    help='batchsize',default=1)
parser.add_argument('--epochnum',dest='epochnum',type = int,
                    help='epoch',default=100)
parser.add_argument('--savemodeldir',dest='savemodeldir',type = str,
                    help='savemodel',default="./savemodel")
opt = parser.parse_args()

print(opt)
if not os.path.exists(opt.savemodeldir):
    os.mkdir(opt.savemodeldir)

if __name__ == '__main__':
    vggfdir = opt.vgg_feature_dir
    audiofdir = opt.audio_feature_dir
    audioformat = opt.audioformat
    T1 = 800
    batch_size = opt.batchsize
    epochnum = opt.epochnum
    #load vggf and audiof
    va_dataset = vggf_audiof_Dataloader(vggfdir,audiofdir,audioformat,T1)  #T1 must 800
    va_dataloader = DataLoader(va_dataset,batch_size = batch_size,pin_memory = True,shuffle=True)
    model = VideoNN(T1).cuda()
    #train
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    for epoch in range(epochnum):
        running_loss = 0
        running_acc = 0
        datacount = 0
        for vggf,audiof in tqdm(va_dataloader):
            torchsize = vggf.shape
            if (torchsize == torch.Size([1,1])):    
                print('not find audiof')
                continue
            datacount += 1
            optimizer.zero_grad()
            vggf = vggf.cuda().float()
            audiof = audiof.cuda().float()
            output = model(vggf).view(1,-1)
            loss = torch.dist(output,audiof,2)
            a= output.cpu().detach().numpy()[0]
            b = audiof.cpu().detach().numpy()[0][0]
            va_stats = stats.pearsonr(a,b)[0]

            if (va_stats > 0.9):
                running_acc += 1

            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            running_loss = running_loss + loss
        running_loss /= datacount
        running_acc /= datacount
        with open (os.path.join(rootdir,'vgg_audio_train_log.txt'),'a') as f:
            f.write(epoch,'/',running_loss,'\n')
        print("[%d/%d] Loss: %.5f, Acc: %.2f"%(epoch+1,epochnum,running_loss,
                                               100*running_acc))
        torch.save(model.state_dict(), os.path.join(opt.savemodeldir,'vgg_au_last_params.pkl'))
        torch.save(model.state_dict(), os.path.join(opt.savemodeldir,'vgg_au_epoch'+str(epoch)+'_params.pkl'))


    
