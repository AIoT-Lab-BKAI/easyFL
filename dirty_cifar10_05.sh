# mkdir fedtask

# # mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_N10_K10_05_100 --dirty_rate 0.5 --noise_magnitude 1 --algorithm fedavg --model cnn --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifarid_10client.json" --num_rounds 99 --num_epochs 5 --proportion 1 --num_epochs_round_0 20 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0 --wandb 1 --result_file_name result_0_5_dirty_rate.json
