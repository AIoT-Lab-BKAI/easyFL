# mkdir fedtask

# # mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=1,2 python main.py  --task dirtycifar10_iid_N10_K10_00_100 --dirty_rate 0.0 0.1 0.2 0.3 0.4 0.5 0.4 0.3 0.2 0.0 --noise_magnitude 1 --algorithm mp_fedavg --model cnn --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_10client.json" --num_rounds 20 --num_epochs 5 --proportion 1 --num_epochs_round_0 5 --batch_size 32 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 1 --wandb 0 --result_file_name dirty_1_data_5k_cnn_5_5_epochs_beta_20_rounds_test_mpfedavg.json --uncertainty 1 --file_log_per_epoch test_mp.csv
