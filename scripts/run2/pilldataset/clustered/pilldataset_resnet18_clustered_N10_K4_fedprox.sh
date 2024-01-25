#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=48:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/pilldataset/$JOB_NAME_$JOB_ID.log
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
LOG_DIR="/home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/pilldataset/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ../2023_CCGRID_Hung/easyFL/benchmark/pilldataset/data/pill3/pill_dataset ${DATA_DIR}

GROUP="pilldataset_resnet18_clustered_N10_K4"
ALG="fedprox"
MODEL="resnet18"
STHR=0.9
EPS=0.5
WANDB=1
ROUND=500
EPOCH_PER_ROUND=5
BATCH=8
PROPOTION=0.4
NUM_THRESH_PER_GPU=1
NUM_GPUS=1
SERVER_GPU_ID=0
TASK="pilldataset_resnet18_clustered_N10_K4"
DATA_IDX_FILE="benchmark/pilldataset/data/syndata3"

cd easyFL

python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG} --eps ${EPS} --sthr ${STHR} --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_path ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} 