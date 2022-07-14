# mkdir fedtask

# # mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=2,3 python main.py  --task dirtycifar10_iid_N10_K10_00_100 --dirty_rate 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 --noise_magnitude 1 --algorithm fedavg --model resnet --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_10client.json" --num_rounds 2 --num_epochs 1 --proportion 1 --num_epochs_round_0 1 --batch_size 64 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0 --wandb 0 --result_file_name dirty_0_data_5k_resnet_5_epochs_beta_1.json --uncertainty 1
