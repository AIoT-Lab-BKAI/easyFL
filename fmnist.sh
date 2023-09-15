
EP=0
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0


EP=1
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha50[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha100[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha500[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0


EP=2
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0


EP=3
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C5_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/5cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha50 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha0.5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha100 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
# cp ../my_wandb.txt ~/../admin/.netrc
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C10_u25_alpha500 --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/10cluster/u0.25_alpha5_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

EP=4
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0

EP=5
SEED=$((EP + 1))
# Episode 1
cp ../my_wandb.txt ~/../admin/.netrc
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python main.py --task fmnist_cnn_journalv1_test01_N100_K10_C0_u25_alpha10[SEC] --ep ${EP} --seed ${SEED} --wandb 1 --model cnn --algorithm journal_v1 --save_agent 1 --load_agent 1 --storage_path "./storage/" --data_folder "../PersonalizedFL/myPFL/benchmark/fmnist/data" --dataidx_filename "fmnist/100client/0cluster/u0.25_alpha0.1_seed1234" --log_folder fedtask  --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_gpus 1 --server_gpu_id 0
