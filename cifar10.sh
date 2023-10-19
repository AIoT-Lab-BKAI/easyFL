#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/journal/
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task cifar10_resnet9_clustered_N100_K10[BACKUP] --learnst 0 --wandb 0 --model cnn --algorithm journal_online --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/cifar10/data" --dataidx_filename "CADIS/cifar10/100client/cluster/" --log_folder fedtask  --num_rounds 1500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
