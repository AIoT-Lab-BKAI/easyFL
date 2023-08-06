# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_cnn_N10_K10 --wandb 1 --model cnn --algorithm proposal_cnn_v2 --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/10client" --log_folder fedtask  --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100client" --log_folder fedtask  --num_rounds 1500 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
