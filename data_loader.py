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
from utils import DataError
from pre_process import letterbox_image

def get_dataset(args, train_ratio = 0.9):
    video_list = os.listdir(args.video_dir)

    size = len(video_list)
    i = int(train_ratio * size) if not args.debug else size - 10
    train_set = MyDataLoader(args, video_list[0:i])
    test_set = MyDataLoader(args, video_list[i:], is_test=True)
    return train_set, test_set

class MyDataLoader(data.Dataset):
    def __init__ (self, args, video_list, is_test=False):
        super(MyDataLoader,self).__init__()
        self.audio_dir          = args.audio_dir
        self.video_dir          = args.video_dir
        self.video_list         = video_list
        self.epoch_size         = args.epoch_size
        self.is_test            = is_test
        self.num_sample         = args.num_sample if not is_test else 30    # only test on first 30 segments

        self.fps                = args.fps
        self.delta              = int(20 / args.fps)    # merge how many frames into one
        self.theta              = args.theta            # onset threshold
        self.use_label          = args.use_label
        self.segment_length     = args.segment_length
        self.dim_video          = args.dim_video

        self.current_ptr = 0
        self.current_sample = self.num_sample
        self.current_complete_video = None
        self.current_complete_label = None
        self.current_total_length = None

    def load_file(self):
        if self.current_sample == self.num_sample:
            self.current_sample = 0 # fetch a new video
        else:   # use the previous one
            return

        if self.current_ptr == len(self.video_list):
            self.current_ptr = 0
            random.shuffle(self.video_list)

        while True:
            try:
                video_name = self.video_list[self.current_ptr]
                self.current_ptr += 1
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
                    strength = strength / torch.max(strength[100:])
                    T = strength.shape[0]
                    r = T % self.delta
                    split = list(strength[0:T-r].split(self.delta))
                    for j in range(len(split)):
                        split[j] = torch.max(split[j])
                    strength = torch.stack(split).view(-1, 1) # (T, 1)
                    label = strength.ge(self.theta) if self.use_label else strength

                self.current_complete_video = video
                self.current_complete_label = label
                self.current_total_length = min(video.shape[0], label.shape[0])
                break

            except FileNotFoundError as e:
                continue
                # raise DataError('MyDataLoader:load_file: ' + str(e))

    def __getitem__(self, index):
        try:
            # sample segment
            self.load_file()    # fetch a new video or use the previous one
            beg = self.segment_length * self.current_sample if self.is_test else \
                  random.randint(0, self.current_total_length - self.segment_length)
            self.current_sample += 1
            # print(self.current_ptr, self.current_sample)
            if beg >= self.current_total_length:
                raise DataError('MyDataLoader:__getitem__: exceed total length')
            end = beg + self.segment_length
            video = self.current_complete_video[beg:end]
            label = self.current_complete_label[beg:end]

            # cut bud segments
            ratio = 1. * torch.sum(label).item() / label.shape[0]
            if ratio < 0.4 or ratio > 0.8:
                raise DataError('MyDataLoader:__getitem__: too many or too few onsets')

            # resize to (1, T, ...)
            video = video.unsqueeze(dim=0)
            label = label.unsqueeze(dim=0)
            # print(self.current_ptr, self.current_sample)
            return video, label

        except DataError as e:
            # print(e)
            zero = torch.tensor([0])
            return zero, zero

    def __len__(self):
        return self.epoch_size
