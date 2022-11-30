mkdir -p results/cifar10

# Pareto N=100, K=10
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K10 --model cnn --algorithm scaffold --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K10 --model cnn --algorithm mp_fedavg --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K10 --model cnn --algorithm mp_proposal --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# Pareto N=100, K=20
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K20 --model cnn --algorithm scaffold --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K20 --model cnn --algorithm mp_fedavg --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K20 --model cnn --algorithm mp_proposal --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# Pareto N=100, K=30
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K30 --model cnn --algorithm scaffold --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K30 --model cnn --algorithm mp_fedavg --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_pareto_N100_K30 --model cnn --algorithm mp_proposal --wandb 0 --data_folder "./benchmark/cifar10/data" --log_folder "results/cifar10" --dataidx_filename "cifar10/100client/CIFAR10_100client_pareto.json" --num_rounds 500 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0