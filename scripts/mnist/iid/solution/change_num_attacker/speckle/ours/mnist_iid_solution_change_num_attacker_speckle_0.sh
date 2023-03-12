#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Cuong_AttackFL/logs/mnist/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

LOG_DIR="/home/aaa10078nj/Federated_Learning/Cuong_AttackFL/logs/mnist/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

# #Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ./easyFL/benchmark/mnist/data ${DATA_DIR}
#LOG_DIR='log_result'
#DATA_DIR='benchmark/mnist/data'
GROUP="dirtymnist_iid_solution_change_num_attacker_speckle"
ALG="mp_fedavg"
MODEL="resnet18"
DIRTY_RATE=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
NOISE_MAGNITUDE=1.0
AGGREGATE='mean'
NOISE_TYPE='speckle'
MALICIOUS_CLIENT=0
ATTACKED_CLASS=(0 1 2 3 4 5 6 7 8 9)
AGG_ALGORITHM="peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)"
OUTSIDE_NOISE="inside"
WANDB=1
ROUND=1000
EPOCH_PER_ROUND=5
BATCH=64
PROPOTION=0.2
NUM_THRESH_PER_GPU=1
NUM_GPUS=1
SERVER_GPU_ID=0
UNCERTAINTY=0
TASK="dirtymnist_iid_solution_change_num_attacker"
IDX_DIR="mnist/iid/mnist_iid_50client_1000data.json"

cd easyFL

python main.py --task ${TASK} --dirty_rate ${DIRTY_RATE[@]} --noise_magnitude ${NOISE_MAGNITUDE} --model ${MODEL} --algorithm ${ALG} --wandb ${WANDB} --data_folder ${DATA_DIR} --log_folder ${LOG_DIR} --dataidx_filename ${IDX_DIR} --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU} --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} --uncertainty ${UNCERTAINTY} --aggregate ${AGGREGATE} --noise_type ${NOISE_TYPE} --num_malicious ${MALICIOUS_CLIENT} --attacked_class ${ATTACKED_CLASS[@]} --agg_algorithm ${AGG_ALGORITHM} --outside_noise ${OUTSIDE_NOISE}
