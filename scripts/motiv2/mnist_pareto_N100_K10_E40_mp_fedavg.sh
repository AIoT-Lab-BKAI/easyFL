CUDA_VISIBLE_DEVICES=0,1 python main.py  --task mnist_pareto_N100_K10_E40  --model cnn  --algorithm mp_fedavg  --wandb 1  --data_folder "./benchmark/mnist/data"  --log_folder "motiv/mnist"  --dataidx_filename "mnist/100client/pareto/MNIST-noniid-pareto_1.json"  --num_rounds 125  --num_epochs 40  --proportion 0.1  --batch_size 16  --num_threads_per_gpu 1   --num_gpus 2  --server_gpu_id 0