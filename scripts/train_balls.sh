#!/bin/bash
python train.py \
  --gpus 0 \
  --n_workers 4 \
  --image_size 128 128 \
  --ckpt_dir $HOME/slowbro/ckpt \
  --dset_name bouncing_balls \
  --evaluate_every 10 \
  --lr_init 1e-3 \
  --lr_decay 1 \
  --n_iters 200000 \
  --hidden_size 128 \
  --batch_size 100 \
  --n_components 4 \
  --stn_scale_prior 4 \
  --ckpt_name 200k
