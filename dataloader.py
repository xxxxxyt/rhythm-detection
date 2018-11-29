import argparse
import os
import cv2
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
#from opt import opt #not complete !!!!!!!!!!!!!!!!!!!!!
from tqdm import tqdm
import json
import numpy as np
import sys
import time
from utils import prep_image_to_tensor,prep_frame
import pickle
rootdir = os.getcwd()

#parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"------------------------------dataloader_________________________________"
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
            T1_to_ori = opt.time*float(orifps)/efps
   
            if ((videolength >= T1_to_ori)and (orifps>=efps)):
                videolist.append(video_name)
            videostream.release()
        except:
            continue
    #save the selectvideolist        
    jsonvideolist = json.dumps(videolist)
    with open (os.path.join(rootdir,'select_'+videodir[-8:]+ '.json'),'w') as f:
        f.write(jsonvideolist)
    
    print('dirlen',len(videolist))
    return videolist

#crop video to opt.time and extract frames
def cropVideo(videopath,efps,T1):
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
            if ((time >= start_label) and (time < end_label)and(label<T1)):#has problem
                crop_list.append(frame)
                time_list.append(time)
                label = label + 1
            if len(crop_list) > T1:
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
    def __init__(self,video_dir,batchSize = 1):
        # initialize the file video stream along with the boolean
        super(RawVideoDataLoader, self).__init__()
        self.T1 = opt.time
        self.efps = opt.efps
        self.video_dir = video_dir
        self.videolist = selectVideo(video_dir,opt.efps)
    
    def __getitem__(self,index):
        try:
            video_name = self.videolist[index]
            video_path = os.path.join(self.video_dir,video_name)
            result_dict = cropVideo(video_path,self.efps,self.T1)
        
            for i in range(len(result_dict['crop_frame'])):
                result_dict['crop_frame'][i] = prep_image_to_tensor(result_dict['crop_frame'][i],
                                                                       opt.indim)        
            return result_dict,video_name
        except:
            return False,False

    def __len__(self):
        return len(self.videolist)

class vggf_audiof_Dataloader(data.Dataset):
    '''
    T1 must 800
    input vggfeaturedir
    output
    '''
    def __init__ (self,vggfdir,audiodir,audio_format,T1 = 800):
        super(vggf_audiof_Dataloader,self).__init__()
        self.vggfdir = vggfdir
        self.vggflist = os.listdir(vggfdir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.audio_format = audio_format  # 1,2
        self.T1 = T1
        self.audiodir = audiodir
        #print(audiodir)

    def __getitem__(self,index):
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
                vggf = self.transform(vggf)
             
            #audio   
            id1 = int(int(identi)/100)+1
            id2 = int(identi)%100
            audioid = 100*id1
            audiof_name = 'dataset_' + group + '_' + str(audioid) + '.pkl'
            with open (os.path.join(self.audiodir,audiof_name),'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                p = u.load()
                find = False
                audio_f = []
                for i in p:
                        
                    if((i[0][0] ==int(group)) and (i[0][1] == int(identi))):
                        find = True
                        for j in range(self.T1):
                            audio_f.append(i[self.audio_format][j])
                        audio_f = np.array(audio_f)
                        audio_f = torch.from_numpy(audio_f)
                        audio_f = audio_f.view(1,-1)
                        break
                if find != True:
                    a = torch.tensor([0])
                    a = a.double()
                    return a,a        
            return vggf,audio_f

        except:
            #return False,Fals
            a = torch.tensor([0])
            a = a.double()
            return a,a

    def __len__(self):
        return len(self.vggflist)










