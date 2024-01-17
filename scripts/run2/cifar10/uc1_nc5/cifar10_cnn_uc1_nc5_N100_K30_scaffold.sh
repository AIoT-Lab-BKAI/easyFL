#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=48:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/cifar10/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
#module load gcc/11.2.0
#Old gcc. Newest support is 12.2.0. See module avail
LD_LIBRARY_PATH=/apps/centos7/gcc/11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/gcc/11.2.0/bin:${PATH}
#module load openmpi/4.1.3
#Old mpi. Use intel mpi instead
LD_LIBRARY_PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/bin:${PATH}
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
#module load python/3.10/3.10.4
#Old python. Newest support is 10.3.10.10. See module avail
LD_LIBRARY_PATH=/apps/centos7/python/3.10.4/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/python/3.10.4/bin:${PATH}

source ~/venv/pytorch1.11+horovod/bin/activate
python --version
LOG_DIR="/home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/cifar10/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ../2023_CCGRID_Hung/easyFL/benchmark/cifar10/data ${DATA_DIR}

GROUP="cifar10_cnn_uc1_nc5_N100_K30"
ALG="scaffold"
MODEL="cnn"
STHR=0.9
WANDB=1
ROUND=3000
EPOCH_PER_ROUND=5
BATCH=8
PROPOTION=0.3
NUM_THRESH_PER_GPU=1
NUM_GPUS=1
SERVER_GPU_ID=0
TASK="cifar10_cnn_uc1_nc5_N100_K30"
DATA_IDX_FILE="journal/cifar10/uc1_nc5/100client"

cd easyFL

python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG} --sthr ${STHR} --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} 