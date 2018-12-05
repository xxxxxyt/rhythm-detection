import os
import json
import sys
import time
import pickle
import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import is_peak, DataError
from pre_process import letterbox_image

def get_dataset(args, train_ratio = 0.9):
    video_list = os.listdir(args.video_dir)
    if args.debug:
        video_list = video_list[0:20]

    size = len(video_list)
    i = int(train_ratio * size)
    train_set = MyDataLoader(args, video_list[0:i])
    test_set = MyDataLoader(args, video_list[i:], is_test=True)
    return train_set, test_set

class MyDataLoader(data.Dataset):
    def __init__ (self, args, video_list, is_test=False):
        super(MyDataLoader,self).__init__()
        self.audio_dir          = args.audio_dir
        self.video_dir          = args.video_dir
        self.video_list         = video_list
        self.is_test            = is_test
        self.num_sample         = args.num_sample if not is_test else 30    # only test on first 30 segments

        self.fps                = args.fps
        self.delta              = int(20 / args.fps)    # merge how many frames into one
        self.theta              = args.theta            # onset threshold
        self.segment_length     = args.segment_length
        self.dim_video          = args.dim_video

        self.current_ptr = None
        self.current_complete_video = None
        self.current_complete_label = None
        self.current_total_length = None

    def load_file(self, ptr):
        try:
            video_name = self.video_list[ptr]
            identi = video_name.split('.')[0].split('_')[-1]

            # video (T, H, W, 3)
            # video_name = 'frames_' + str(identi) + '.pkl'
            with open(os.path.join(self.video_dir, video_name), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'iso-8859-1'
                video = u.load().float() # (T, H, W, c)
            # label (T, 1)
            audio_name = 'feature_3_' + str(identi) + '.pkl'
            with open (os.path.join(self.audio_dir, audio_name), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                strength = torch.tensor(u.load()).float()[:,0] # (T,)
                T = strength.shape[0]
                r = T % self.delta
                split = list(strength[0:T-r].split(self.delta))
                for j in range(len(split)):
                    split[j] = torch.max(split[j])
                strength = torch.stack(split)
                T = strength.shape[0]
                label = torch.zeros(T, 1).long()    # (T, 1)
                for t in range(T):
                    if strength[t] > self.theta and is_peak(t, strength):
                        label[t] = 1
            
            self.current_ptr = ptr
            self.current_complete_video = video
            self.current_complete_label = label
            self.current_total_length = min(video.shape[0], label.shape[0])

        except FileNotFoundError as e:
            raise DataError('MyDataLoader:load_file: ' + str(e))

    def __getitem__(self, index):
        if index == 0 and not self.is_test:
            random.shuffle(self.video_list)
        
        try:
            ptr = int(index / self.num_sample)
            sample = index % self.num_sample
            if sample == 0: # load a new video
                self.load_file(ptr)
            if ptr != self.current_ptr:
                raise DataError('MyDataLoader:__getitem__: no such file')

            # sample segment
            if self.is_test: # test: traverse segments
                beg = self.segment_length * sample
                if beg >= self.current_total_length:
                    raise DataError('MyDataLoader:__getitem__: exceed total length')
            else: # train: sample segments
                beg = random.randint(0, self.current_total_length - self.segment_length)
            end = beg + self.segment_length
            video = self.current_complete_video[beg:end]
            label = self.current_complete_label[beg:end]

            # resize to (1, T, ...)
            video = video.unsqueeze(dim=0)
            label = label.unsqueeze(dim=0)
            return video, label

        except DataError as e:
            # print(e)
            zero = torch.tensor([0])
            return zero, zero

    def __len__(self):
        return len(self.video_list) * self.num_sample
