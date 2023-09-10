#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/journal/

# cp ../my_wandb.txt ~/../admin/.netrc
EP="infer"
# Episode 1
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task mnist_cnn_journalv1_test01_N100_K10_C4_u20_alpha50  --wandb 1 --model cnn --algorithm cadis --kd_fct 1 --sthr 0.975 --data_folder "../PersonalizedFL/myPFL/benchmark/mnist/data" --dataidx_filename "mnist/100client/0cluster/u0.2_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task mnist_cnn_journalv1_test01_N100_K10_C4_u20_alpha50  --wandb 1 --model cnn --algorithm scaffold --data_folder "../PersonalizedFL/myPFL/benchmark/mnist/data" --dataidx_filename "mnist/100client/0cluster/u0.2_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task mnist_cnn_journalv1_test01_N100_K10_C4_u20_alpha50 --ep ${EP} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 0 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/mnist/data" --dataidx_filename "mnist/100client/0cluster/u0.2_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
