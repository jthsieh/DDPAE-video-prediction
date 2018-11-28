#!/bin/bash
python test.py \
  --gpus 0 \
  --n_workers 4 \
  --batch_size 16 \
  --dset_name bouncing_balls \
  --ckpt_dir $HOME/slowbro/ckpt \
  --log_every 5 \
  --hidden_size 128 \
  --n_components 4 \
  --save_visuals 0 \
  --save_results 1 \
  --ckpt_name crop_NC4_lr1.0e-03_bt100_200k
