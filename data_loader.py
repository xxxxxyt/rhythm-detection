import os
import json
import sys
import time
import pickle
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import prep_image_to_tensor, prep_frame, is_peak

root_dir = os.getcwd()

#parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"------------------------------dataloader------------------------------"
#parser.add_argument('--videodir',dest='inputpath',type = str,
#                    help='video-directory',default="")
#parser.add_argument('--time',dest = 'time',default=4500,type=int,
#                     help='choose the unify video length to train and test')
#parser.add_argument('--inpdim',dest = 'inp_dim',default=224,type=int,
#                     help = 'input dim for vgg19')
#parser.add_argument('--efps',dest = 'efps',default=20,type=int,
#                     help='the count of extract frame per second')
#parser.add_argument('--indim',dest = 'indim',default=224,type=int,
#                     help='imgsize input vgg')

#opt = parser.parse_args()

#select video more than opt.time
def selectVideo(videodir,efps):
    '''
    select the time > opt.time
    '''
    videoid = os.listdir(videodir)
    videolist = []
    print('select the video')
    for video_name in tqdm(videoid):
        try:
            videostream = cv2.VideoCapture(os.path.join(videodir,video_name))
            videolength = videostream.get(cv2.CAP_PROP_FRAME_COUNT)
            orifps = videostream.get(cv2.CAP_PROP_FPS)    
            if not videolength:
                raise IOError
            segment_length_to_ori = opt.time*float(orifps)/efps
   
            if ((videolength >= segment_length_to_ori)and (orifps>=efps)):
                videolist.append(video_name)
            videostream.release()
        except:
            continue
    #save the selectvideolist        
    jsonvideolist = json.dumps(videolist)
    with open (os.path.join(root_dir,'select_'+videodir[-8:]+ '.json'),'w') as f:
        f.write(jsonvideolist)
    
    print('dirlen',len(videolist))
    return videolist

#crop video to opt.time and extract frames
def cropVideo(videopath,efps,segment_length):
    '''
    input video path    and   efs
    output crop frame dict  type(dict) contain frame_list and time_list
    if wrong return 1
    '''
    try:
        videostream = cv2.VideoCapture(videopath)
        orilen = int(videostream.get(cv2.CAP_PROP_FRAME_COUNT))
        orifps = videostream.get(cv2.CAP_PROP_FPS)
        if (orifps < efps):
            print('orifps < efps')
            return 1
        crop_list = []
        time_list = []
        time_per_frame = 1/orifps
        frame_interval = 1/efps
        time = 0
        label = 0 #use for extract frame
        for i in range(orilen):
            (grabbed,frame) = videostream.read()
            start_label = label*frame_interval
            end_label = (label+1)*frame_interval
            if ((time >= start_label) and (time < end_label)and(label<segment_length)):#has problem
                crop_list.append(frame)
                time_list.append(time)
                label = label + 1
            if len(crop_list) > segment_length:
                break
            time = time + time_per_frame
        result_dict = {}
        result_dict['crop_frame'] = crop_list
        result_dict['time_list'] = time_list     

        videostream.release()
        return result_dict
    except :
        print('open wrong',videopath)
        return 1
    

class RawVideoDataLoader(data.Dataset):
    '''
    input video_dir
    output each video extract frame(tensor) and time list
    '''
    def __init__(self, video_dir, batchSize = 1):
        # initialize the file video stream along with the boolean
        super(RawVideoDataLoader, self).__init__()
        self.segment_length = opt.time
        self.efps = opt.efps
        self.video_dir = video_dir
        self.videolist = selectVideo(video_dir,opt.efps)
    
    def __getitem__(self, index):
        try:
            video_name = self.videolist[index]
            video_path = os.path.join(self.video_dir,video_name)
            result_dict = cropVideo(video_path,self.efps,self.segment_length)
        
            for i in range(len(result_dict['crop_frame'])):
                result_dict['crop_frame'][i] = prep_image_to_tensor(result_dict['crop_frame'][i],
                                                                       opt.indim)        
            return result_dict,video_name
        except:
            return False,False

    def __len__(self):
        return len(self.videolist)

class MyDataLoader(data.Dataset):
    def __init__ (self, args):
        super(MyDataLoader,self).__init__()
        self.audio_dir          = args.audio_dir
        self.video_dir          = args.video_dir
        self.fps                = args.fps
        self.delta              = 20 / args.fps         # merge how many frames into one
        self.theta              = args.theta            # onset threshold
        self.segment_length     = args.segment_length

    def __getitem__(self, index):
        try:
            vggf_name = self.vggflist[index]
            vgg_id = vggf_name.split('.')[0]
            group = vgg_id.split('_')[0]
            identi = vgg_id.split('_')[1]

            # video
            video = None
            # TODO: video shape of (T, H, W) or what you feed into VGG network (if so, please comment for me)
            # T is the length under args.fps
                
            # label
            audiof_name = 'feature_' + group + '_' + str(identi) + '.pkl'
            with open (os.path.join(self.audio_dir, audiof_name), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                strength = torch.Tensor(u.load()).float()[:,0] # (T,)
                T = strength.shape[0]
                r = T % self.delta
                split = np.array_split(strength[0:T-r], T/self.delta)
                for j in range(len(split)):
                    split[j] = torch.max(split[j])
                strength = torch.stack(split)
                T = strength.shape[0]
                label = torch.zeros(T, 1).long()
                for t in range(T):
                    if strength[t] > self.theta and is_peak(t, strength):
                        label[t] = 1

            T = min([video.shape[0], label.shape[0]])
            r = T % self.segment_length
            video = video[0:T-r].reshape() # TODO:
            label = label[0:T-r].reshape(-1, self.segment_length, 1) # (n, T, D)

        except:
            zero = torch.Tensor([0])
            video, label = zero, zero

        finally:
            return video, label

    def __len__(self):
        return len(self.vggflist)

class vggf_audiof_Dataloader(data.Dataset):
    '''
    segment_length must 800
    input vggfeaturedir
    output
    '''
    def __init__ (self, args):
        super(vggf_audiof_Dataloader,self).__init__()
        self.audio_dir = args.audio_dir
        self.vggfdir = args.vggf_dir
        self.vggflist = os.listdir(args.vggf_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.segment_length = args.segment_length
        self.hint_length = args.hint_length
        self.multi_to_one = args.multi_to_one
        self.fps = args.fps
        self.theta = args.theta

    def __getitem__(self, index):
        try:
            vggf_name = self.vggflist[index]
            vgg_id = vggf_name.split('.')[0]
            group = vgg_id.split('_')[0]
            identi = vgg_id.split('_')[1]
            
            #vggf
            vggf_path = os.path.join(self.vggfdir,vggf_name)
            with open (vggf_path,'r') as f:
                vggf_t = json.load(f)
                vggf = vggf_t['vggfeaturemap']
                vggf = np.array(vggf)
                vggf = self.transform(vggf).squeeze().transpose(0, 1).float() # (T, D)
                if not self.multi_to_one: # multi-to-multi
                    T, D = vggf.shape
                    r = T % self.segment_length
                    vggf = vggf[0:T-r].reshape(-1, self.segment_length, D) # (n, T, D)
                
            #audio
            audiof_name = 'feature_' + group + '_' + str(identi) + '.pkl'
            with open (os.path.join(self.audio_dir, audiof_name), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                audiof = torch.Tensor(u.load()).float() # (T, D)
                if not self.multi_to_one: # multi-to-multi
                    T, D = audiof.shape
                    r = T % self.segment_length
                    audiof = audiof[0:T-r].reshape(-1, self.segment_length, D) # (n, T, D)

            if self.multi_to_one:
                vggf_list = []
                audiof_list = []
                T = min([vggf.shape[0], audiof.shape[0]])
                for t in range(self.hint_length, T - self.hint_length):
                    vggf_ = vggf[t-self.hint_length:t+self.hint_length+1,:] # (T, D)
                    audiof_ = audiof[t-self.hint_length:t+self.hint_length+1,:] # (T, D)
                    vggf_list += [vggf_]
                    audiof_list += [audiof_]
                vggf = torch.stack(vggf_list, dim=0)
                audiof = torch.stack(audiof_list, dim=0)
            else: # multi-to-multi format
                n = min([vggf.shape[0], audiof.shape[0]])
                vggf = vggf[0:n]
                audiof = audiof[0:n]

        except:
            zero = torch.Tensor([0])
            vggf, audiof = zero, zero

        finally:
            return vggf, audiof

    def __len__(self):
        return len(self.vggflist)
