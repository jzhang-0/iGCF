#!/usr/bin/env bash
# ~/pyenv3/bin/python ./launch.py -data_path ./data/data/ -environment env -T 40 -ST [5,10,20,40] -agent Train -FA FA -latent_factor 50 \
# -learning_rate 0.001 -training_epoch 3000 -seed 145 -gpu_no 0 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False

# python src/script/nicf_launch.py --datan ml-100k -p 0.5 --task "coldstart"  -T 40 -ST [5,10,20,40] -agent Train -FA FA -latent_factor 50 \
# -learning_rate 0.001 -training_epoch 3000 -seed 123 -gpu_no 0 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False -environment env

# conda init
# conda activate nicf
python src/script/nicf_launch.py --datan ml-100k -p 0.5 --task "coldstart"  -T 120 -ST [5,10,20,40,60,120] -agent Train -FA FA -latent_factor 50 \
-learning_rate 0.001 -training_epoch 3000 -seed 123 -gpu_no 6 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False -environment env
