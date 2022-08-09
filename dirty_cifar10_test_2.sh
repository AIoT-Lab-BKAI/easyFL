# mkdir fedtask

# # mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=1 python main.py  --task dirtycifar10_iid_N10_K10_00_100 --dirty_rate 0.0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4 0.5 --noise_magnitude 1 --algorithm fedavg --model resnet --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_10client.json" --num_rounds 50 --num_epochs 5 --proportion 1 --num_epochs_round_0 5 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0 --wandb 0 --result_file_name resnettest.json --uncertainty 1 --file_log_per_epoch resnettest.csv
