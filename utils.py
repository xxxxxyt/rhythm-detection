import os
import sys
import json
import cv2
from PIL import Image, ImageDraw
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class DataError(Exception):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

def collate_fn(batch):
    video_list = []
    label_list = []
    for video, label in batch:
        if len(label.shape) == 1:
            continue
        video_list += [video]
        label_list += [label]
    if len(label_list) < 1:
        zero = torch.tensor([0])
        return zero, zero
    video = torch.cat(video_list)
    label = torch.cat(label_list)
    return video, label

# def is_peak(t, x):
#     """
#     t:  float
#     x:  (T,)
#     """
#     if t == 0:
#         return x[t] > x[t + 1]
#     elif t == x.shape[0] - 1:
#         return x[t] > x[t - 1]
#     else:
#         return x[t] > x[t + 1] and x[t] > x[t - 1]
        