1.vgg_net.py       extract frame video to pass by it  ----> 1000dim   #complete
2.nn.py ---------> Videonn:for feature extracted by vgg (1000dim)  #complete
        ---------> Audionn: for feature extracted by xyt           #complete
        ---------> FeatureVideonn:for feature extracted by   pose\scene\faceEmotion #complete
3.dataloader.py ---------> RawVideoDataLoader: load raw videodata #complete
           ---------> 3FeatureVideoDataLoader: load extracted feature of video
           ---------> AudioDataLoader :load audio dataloader      # yutong work for it
           ---------> vggfeaturevideoDataloader: load vgg feature of video
           ---------> batcher #
           ---------> selectVideo :select the video length more than opt.time #complete
           ---------> ##cropVideo: crop video to fixed length #
4.utils.py ---------> preq 
4.opt.py 
5.demo_raw.py
6.demo_feature.py
7.train_raw.py      --videodir   --time(default = 300)
8.train_feature.py


python train.py --vggfeaturedir  --audiofeaturedir  --audioformat 1 --savemodeldir  



audio feature 1 (0,1)regre
              2 classify
