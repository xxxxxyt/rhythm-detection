# Visual Rhythm Detection

## TODO

- data_loader.py: load video
- models.py: self.vgg_net
- train.py: load pre-trained params for vgg_net

## Run

```
python train.py \
    --audio_dir ../audio_3_split \
    --video_dir ... \
    --debug 1 \
    --batch_size 1 \
    --eval_every 1 \
    --fps 4 \
    --model EndToEndCRF
```