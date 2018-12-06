import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

# video (T, H, W, 3)
# video_name = 'frames_' + str(identi) + '.pkl'
# with open(os.path.join(self.video_dir, video_name), 'rb') as f:
#     u = pickle._Unpickler(f)
#     u.encoding = 'iso-8859-1'
#     video = u.load().float() # (T, H, W, c)
# label (T, 1)

delta = 5
theta = 0.3

use_label = 0
down_sample = 1

audio_dir = '../audio_3_split'
audio_name = 'feature_3_5128.pkl'
with open (os.path.join(audio_dir, audio_name), 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    u = u.load()
    strength = torch.tensor(u).float()[:,0] # (T,)
    strength = strength / torch.max(strength[100:])
    if down_sample:
        T = strength.shape[0]
        r = T % delta
        split = list(strength[0:T-r].split(delta))
        for j in range(len(split)):
            split[j] = torch.max(split[j])
        strength = torch.stack(split).view(-1, 1) # (T, 1)
    label = strength.ge(theta) if use_label else strength
    label = np.array(label)

    beg = 0 if down_sample else 0
    end = 600 if down_sample else 3000

    plt.figure()
    plt.plot(list(range(beg, end)), label[beg:end])
    plt.plot(list(range(beg, end)), [0.3]*(end-beg))
    plt.show()