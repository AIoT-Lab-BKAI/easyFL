GROUP="emnist_sparse_N500_K10_E4"
ALG="mp_feddyn"
MODEL="cnn"
WANDB=1
ROUND=1000
EPOCH_PER_ROUND=4
BATCH=8
PROPOTION=0.02
NUM_THRESH_PER_GPU=1
NUM_GPUS=2
SERVER_GPU_ID=0
TASK="emnist_sparse_N500_K10_E4"
DATA_IDX_FILE="emnist/sparse/500client/emnist_sparse.json"

cd easyFL

python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG}  --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} 