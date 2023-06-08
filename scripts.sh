# CLUSTERED
mkdir fedtask
# mnist_clustered_N100_K20 - quantitative
CUDA_VISIBLE_DEVICES=0 python main.py --task mnist_clustered_N100_K10 --wandb 0 --model cnn --algorithm proposal --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0