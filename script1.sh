# SPARSE
mkdir fedtask

# CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --wandb 1 --model cnn --algorithm proposal_flatten --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 300 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
# CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --wandb 1 --model cnn --algorithm proposal_cnn_v2 --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 300 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferA --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientA" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferB --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientB" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferC --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientC" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferA --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientA" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferB --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientB" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1,0 python main.py --task cifar10_cnn_N100_K10 --wandb 1 --model cnn --algorithm proposal_dnn --buffer bufferC --data_folder "./benchmark/cifar10/data" --dataidx_filename "cifar10/sparse/100clientC" --log_folder fedtask  --num_rounds 200 --num_epochs 5 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
