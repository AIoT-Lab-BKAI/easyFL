# CUDA_VISIBLE_DEVICES=2 python main.py --task cifar10_cnn_pareto_N100_K10 --model cnn --algorithm journal_v3 --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "CADIS/cifar10/100client/pareto" --ep 'infer' --save_agent 0 --load_agent 0 --storage_path "./storage/" --num_rounds 3000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task cifar10_cnn_PA_0.5_N100_K10 --model cnn --algorithm journal_v5 --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "journal/cifar10/pareto_0.5/100client" --ep 'infer' --save_agent 0 --load_agent 0 --storage_path "./storage/" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task cifar10_cnn_BC_NC5_N100_K10 --model cnn --algorithm journal_v5 --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "journal/cifar10/uc1_nc5/100client" --ep 'infer' --save_agent 0 --load_agent 0 --storage_path "./storage/" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task cifar10_cnn_UC_NC5_N100_K10 --model cnn --algorithm journal_v5 --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "journal/cifar10/uc4_nc5/100client" --ep 'infer' --save_agent 0 --load_agent 0 --storage_path "./storage/" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
