# Visual Rhythm Detection

## TODO

- data_loader.py: load video
- models.py: self.vgg_net
- train.py: load pre-trained params for vgg_net

## Run

```
# gpu17-1
python train.py \
    --audio_dir ../audio_3_split \
    --video_dir ../video_3_frames_4fps \
    --debug 1 \
    --device 1 \
    --batch_size 1 \
    --eval_every 1 \
    --fps 4 \
    --model EndToEndCRF

# gpu17-2
python train.py \
    --audio_dir ../audio_3_split \
    --video_dir ../video_3_frames_4fps \
    --device 2 \
    --batch_size 1 \
    --eval_every 1 \
    --fps 4 \
    --model EndToEndCRF
```