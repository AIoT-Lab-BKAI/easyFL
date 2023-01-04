mkdir fedtask

CUDA_VISIBLE_DEVICES=1,2 python main.py --task mnist_sparse_N100_K10 --wandb 0 --model mlp --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/sparse/100client/mnist_sparse.json --num_rounds 250 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,2 python main.py --task mnist_sparse_N100_K10 --wandb 0 --model mlp --algorithm scaffold --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/sparse/100client/mnist_sparse.json --num_rounds 250 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,2 python main.py --task mnist_sparse_N100_K10 --wandb 0 --model mlp --algorithm mp_proposal --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/sparse/100client/mnist_sparse.json --num_rounds 250 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

