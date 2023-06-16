# conda activate nicf
python src/script/nicf_launch.py --datan EachMovie -p 0.5 --task "coldstart"  -T 120 -ST [5,10,20,40,60,120] -agent Train -FA FA -latent_factor 50 \
-learning_rate 0.001 -training_epoch 3000 -seed 123 -gpu_no 2 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False -environment env
