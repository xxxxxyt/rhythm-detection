import argparse
import json
import os
import random
from tqdm import tqdm

import cv2
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader, Subset

import models
from data_loader import get_dataset
from utils import collate_fn, is_peak, DataError

parser = argparse.ArgumentParser(description='PyTorch Training')
rootdir = os.getcwd()
# environment
parser.add_argument('--audio_dir',  type=str,   default='')
parser.add_argument('--video_dir',  type=str,   default='')
# parser.add_argument('--vggf_dir',   type=str,   default='')
parser.add_argument('--save_dir',   type=str,   default='')
parser.add_argument('--debug',      type=int,   default=0)
parser.add_argument('--device',     type=int,   default=-1)
# parser.add_argument('--infer_answer', type=int, default=0)
# train
parser.add_argument('--batch_size', type=int,   default=10,     help='batchsize')
parser.add_argument('--num_epoch',  type=int,   default=10000,  help='epoch')
parser.add_argument('--eval_every', type=int,   default=1,      help='evaluate how many every epoch')
parser.add_argument('--save_every', type=int,   default=50,     help='save model how many every epoch')
# data
parser.add_argument('--fps',        type=int,   default=20)
parser.add_argument('--theta',      type=float, default=0.1,    help='onset threshold')
parser.add_argument('--num_sample', type=int,   default=4,      help='sample how many segments in a video during one epoch')
# parser.add_argument('--mask', type=int, default=0, help='mask some negative data points')
# parser.add_argument('--label', type=str, default='strength')
# model
parser.add_argument('--segment_length', type=int, default=40)
parser.add_argument('--dim_feature',    type=int, default=1000)
parser.add_argument('--dim_video',      type=int, default=224)
parser.add_argument('--model',          type=str, default='RNN')
parser.add_argument('--use_crf',        type=int, default=0)
# parser.add_argument('--shift_with_attention', type=int, default=0)

args = parser.parse_args()
if len(args.save_dir) == 0:
    if args.debug:
        args.save_dir = 'debug'
    else:
        args.save_dir = 'train'
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
else:
    print('save_dir already exist, input YES to comfirm')
    s = input()
    if 'YES' not in s:
        assert False

if 'CRF' in args.model:
    args.use_crf = 1

######################################################

def handle_batch(args, batch):
    video, label = batch
    if len(label.shape) == 1:
        raise DataError('handle_batch: bad batch')
    with torch.no_grad():
        video = video.cuda()    # (n, T, H, W)
        label = label.cuda()    # (n, T, 1) long
    output = model(video)   # (n, T, 1) long
    return video, label, output

def train(model, train_set, test_set, args):
    data_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn)

    adam = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    for epoch in range(args.num_epoch):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(data_loader):
            try:
                video, label, output = handle_batch(args, batch)

                adam.zero_grad()
                if args.use_crf:
                    loss = model.loss(video, label)
                else: # compute loss by prediction
                    loss = torch.nn.functional.binary_cross_entropy(output, label)
                
                loss.backward()
                adam.step()
                epoch_loss += loss.item()
            except DataError as e:
                # print(e)
                continue

        print(output.reshape(-1))
        print(label.reshape(-1))

        report = 'epoch[%d/%d] Loss: %.5f' % (epoch+1, args.num_epoch, epoch_loss)
        with open (os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
            f.write(report + '\n')
        print(report)

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_'+str(epoch)+'_params.pkl'))
        if epoch % args.eval_every == 0:
            evaluate(model, test_set, args)

def evaluate(model, test_set, args):
    data_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn)
    cnt_match, cnt_onset, cnt_pred = 0, 0, 0
    model.eval()
    for batch in tqdm(data_loader):
        try:
            video, label, output = handle_batch(args, batch)
            for n in range(label.shape[0]):
                for t in range(label.shape[1]):
                    flag_label = True if label[n,t,0] > args.theta else False
                    flag_pred = True if output[n,t,0] > args.theta and \
                        (args.use_crf or is_peak(t, output[n,:,0])) else False
                    cnt_onset += flag_label
                    cnt_pred += flag_pred
                    cnt_match += flag_label and flag_pred
        except DataError as e:
            # print(e)
            continue
    
    print(output.reshape(-1))
    print(label.reshape(-1))

    prec   = 1. * cnt_match / cnt_pred if cnt_pred > 0 else 0
    recall = 1. * cnt_match / cnt_onset
    f1 = 2. * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    report = 'Evaluation: F1 %.4f (%.4f %i/%i, %.4f %i/%i)' % (f1, prec, cnt_match, cnt_pred, recall, cnt_match, cnt_onset)
    with open (os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
        f.write(report + '\n')
    print(report)

if __name__ == '__main__':

    # prepare dataset
    train_set, test_set = get_dataset(args)

    # define model
    Model = getattr(models, args.model)
    if args.device > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
        model = Model(args).cuda()

    # train
    train(model, train_set, test_set, args)
