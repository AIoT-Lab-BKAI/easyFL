import os
import textwrap


visible_cudas = [0, 1]
cudas = ",".join([str(i) for i in visible_cudas])
task_file = "main.py"

dataset = "pilldataset"
sthr = 0.9

# dataset_types = ["dirichlet_0.5", "pareto2_1"]
dataset_types = ["clustered"]
# dataset_types = ["uc1_nc5", "uc4_nc5"]
model = "resnet18"
# model = "cnn"

# config parameters
N = 10
rate = 0.5
K = int(N*rate)
E = 5
batch_size = 8
num_round = 500
epsilon = 0.5


# algos = ["singleset", "cadis", "fedavg"]
# algos = ["scaffold", "journal_v4_pill"]
algos = ["fedavg", "fedprox", "fedfa", "cadis", "journal_v4_pill"]

data_folder = f"./benchmark/{dataset}/data"
log_folder = f"motiv/{dataset}"



header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=96:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/pilldataset/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
#module load gcc/11.2.0\n\
#Old gcc. Newest support is 12.2.0. See module avail\n\
LD_LIBRARY_PATH=/apps/centos7/gcc/11.2.0/lib:${LD_LIBRARY_PATH}\n\
PATH=/apps/centos7/gcc/11.2.0/bin:${PATH}\n\
#module load openmpi/4.1.3\n\
#Old mpi. Use intel mpi instead\n\
LD_LIBRARY_PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/lib:${LD_LIBRARY_PATH}\n\
PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/bin:${PATH}\n\
module load cuda/11.5/11.5.2\n\
module load cudnn/8.3/8.3.3\n\
module load nccl/2.11/2.11.4-1\n\
#module load python/3.10/3.10.4\n\
#Old python. Newest support is 10.3.10.10. See module avail\n\
LD_LIBRARY_PATH=/apps/centos7/python/3.10.4/lib:${LD_LIBRARY_PATH}\n\
PATH=/apps/centos7/python/3.10.4/bin:${PATH}\n\n\
source ~/venv/pytorch1.11+horovod/bin/activate\n\
python --version\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/Ha_CADIS_FEDRL/logs/pilldataset/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ../2023_CCGRID_Hung/easyFL/benchmark/pilldataset/data/pill3/pill_dataset ${DATA_DIR}\n\n\
"

for dataset_type in dataset_types:
    
    task_name = f"{dataset}_{model}_{dataset_type}_N{N}_K{K}"
    
    for algo in algos:
    
        command = f"\
        GROUP=\"{task_name}\"\n\
        ALG=\"{algo}\"\n\
        MODEL=\"{model}\"\n\
        STHR={sthr}\n\
        EPS={epsilon}\n\
        WANDB=1\n\
        ROUND={num_round}\n\
        EPOCH_PER_ROUND={E}\n\
        BATCH={batch_size}\n\
        PROPOTION={rate}\n\
        NUM_THRESH_PER_GPU=1\n\
        NUM_GPUS=1\n\
        SERVER_GPU_ID=0\n\
        TASK=\"{task_name}\"\n\
        DATA_IDX_FILE=\"benchmark/pilldataset/data/syndata3\"\n\n\
        cd easyFL\n\n\
        "

        # task_name = f"{dataset}_{dataset_type}_N{N}_K{K}_E{E}"
        # command = formated_command.format(
        #     task_name, algo, model, 1000, E, batch_size, K/N, task_name, dataset_type, N
        # )
            
        body_text = "python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG} --eps ${EPS} --sthr ${STHR} --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_path ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} "

        dir_path = f"./run2/{dataset}/{dataset_type}/"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        file = open(dir_path + f"{task_name}_{algo}.sh", "w")
        file.write(header_text + textwrap.dedent(command) + body_text)
        file.close()