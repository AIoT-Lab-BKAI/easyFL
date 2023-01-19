mkdir fedtask

CUDA_VISIBLE_DEVICES=0,3 python main.py --task mnist_dir_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_proposal --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename "mnist/dirichlet/dir_1_sparse/100client" --num_rounds 10 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,3 python main.py --task mnist_dir_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename "mnist/dirichlet/dir_1_sparse/100client" --num_rounds 10 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,3 python main.py --task mnist_dir_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename "mnist/dirichlet/dir_1_sparse/100client" --num_rounds 10 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,3 python main.py --task mnist_dir_sparse_N100_K10 --wandb 0 --model cnn --algorithm scaffold --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename "mnist/dirichlet/dir_1_sparse/100client" --num_rounds 10 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
