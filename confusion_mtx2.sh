CUDA_VISIBLE_DEVICES=2 python main.py --task mnist_cnn_uc5_nc5_N10_K10 --model cnn --algorithm journal_v4 --eps 1 --kd_fct 1.0 --wandb 1 --data_folder "./benchmark/mnist/data" --log_folder "fedtask" --dataidx_filename "journal/mnist/uc5_nc5/10client" --num_rounds 100 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 1 --server_gpu_id 0