# mkdir fedtask

# # mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,2 python main.py  --task dirtycifar10_iid_N10_K10_00_100 --dirty_rate 0.0 0.1 0.2 0.3 0.4 0.5 0.4 0.3 0.2 0.0 --noise_magnitude 1 --algorithm mp_fedavg --model resnet --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_10client.json" --num_rounds 20 --num_epochs 100 --proportion 1 --num_epochs_round_0 100 --batch_size 64 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0 --wandb 1 --result_file_name mp_dirty_1_data_5k_resnet_100_epochs_beta.json --uncertainty 1
