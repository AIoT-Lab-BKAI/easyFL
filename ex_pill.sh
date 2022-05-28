mkdir fedtask

CUDA_VISIBLE_DEVICES=0,1 python main.py --task pilldataset_clustered_N100_K10 --model resnet18 --algorithm mp_fedavg --data_folder ./dataset_idx/pill_dataset --log_folder fedtask  --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task pilldataset_clustered_N100_K10 --model resnet18 --algorithm mp_fedprox --data_folder ./dataset_idx/pill_dataset --log_folder fedtask  --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task pilldataset_clustered_N100_K10 --model resnet18 --algorithm fedtest --data_folder ./dataset_idx/pill_dataset --log_folder fedtask  --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
