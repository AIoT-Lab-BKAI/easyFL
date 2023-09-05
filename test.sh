cp ../my_wandb.txt ~/../admin/.netrc

CUDA_VISIBLE_DEVICES=0 python main.py --task fmnist_cnn_journalv1_test01_N100_K10 --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 0 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 100 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
