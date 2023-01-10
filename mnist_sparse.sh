mkdir fedtask

CUDA_VISIBLE_DEVICES=1 python main.py --task mnist_dir_sparse_N100_K10 --wandb 0 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar10/data --log_folder fedtask --dataidx_filename "mnist/dirichlet/dir_1_sparse/100client" --num_rounds 250 --num_epochs 8 --proportion 0.1 --batch_size 4 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
