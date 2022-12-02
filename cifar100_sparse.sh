# CIFAR100

# N=100, K=10
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 100 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K10 --wandb 0 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 100 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_proposal --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 100 --num_epochs 8 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# N=100, K=20
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K20 --wandb 0 --model resnet9 --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K20 --wandb 0 --model resnet9 --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K20 --wandb 0 --model resnet9 --algorithm mp_proposal --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# # N=100, K=30
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K30 --wandb 0 --model resnet9 --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K30 --wandb 0 --model resnet9 --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K30 --wandb 0 --model resnet9 --algorithm mp_proposal --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# # N=100, K=40
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K40 --wandb 0 --model resnet9 --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K40 --wandb 0 --model resnet9 --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K40 --wandb 0 --model resnet9 --algorithm mp_proposal --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# # N=100, K=50
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K50 --wandb 0 --model resnet9 --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K50 --wandb 0 --model resnet9 --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_sparse_N100_K50 --wandb 0 --model resnet9 --algorithm mp_proposal --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar100/sparse/100client/cifar100_sparse.json --num_rounds 1000 --num_epochs 8 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
