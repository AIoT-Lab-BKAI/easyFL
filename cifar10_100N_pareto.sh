CUDA_VISIBLE_DEVICES=1 python main.py --task cifar10_cnn_pareto_N100_K10 --model cnn --algorithm cadis --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "CADIS/cifar10/100client/pareto" --num_rounds 3000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1 python main.py --task cifar10_cnn_pareto_N100_K10 --model cnn --algorithm scaffold --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "CADIS/cifar10/100client/pareto" --num_rounds 3000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=1 python main.py --task cifar10_cnn_pareto_N100_K10 --model cnn --algorithm mp_fedavg --wandb 1 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "CADIS/cifar10/100client/pareto" --num_rounds 3000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0
