import os

visible_cudas = [0, 1]
cudas = ",".join([str(i) for i in visible_cudas])
task_file = "main.py"

dataset = "mnist"
sthr = 0.975

dataset_types = ["dirichlet_0.3", "pareto2_0.3"]
# model = "resnet9"
model = "cnn"

# config parameters
N = 100
rate = 0.1
K = int(N*rate)
E = 5
batch_size = 8
num_round = 3000


algos = ["singleset", "cadis", "fedavg"]
# algos = ["singleset", "scaffold", "fedavg", "fedprox", "fedfa", "cadis"]

data_folder = f"./benchmark/{dataset}/data"
log_folder = f"motiv/{dataset}"



header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=36:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/Journal_CADIS_FEDRL/logs/cifar100/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
module load gcc/11.2.0\n\
module load openmpi/4.1.3\n\
module load cuda/11.5/11.5.2\n\
module load cudnn/8.3/8.3.3\n\
module load nccl/2.11/2.11.4-1\n\
module load python/3.10/3.10.4\n\
source ~/venv/pytorch1.11+horovod/bin/activate\n\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/Journal_CADIS_FEDRL/logs/cifar100/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ./easyFL/benchmark/cifar100/data ${DATA_DIR}\n\n\
"

for dataset_type in dataset_types:
    
    task_name = f"{dataset}_{model}_{dataset_type}_N{N}_K{K}"
    
    for algo in algos:
    
        command = f"\
        GROUP=\"{task_name}\"\n\
        ALG=\"{algo}\"\n\
        MODEL=\"{model}\"\n\
        STHR={sthr}\n\
        WANDB=1\n\
        ROUND={num_round}\n\
        EPOCH_PER_ROUND={E}\n\
        BATCH={batch_size}\n\
        PROPOTION={rate}\n\
        NUM_THRESH_PER_GPU=1\n\
        NUM_GPUS=1\n\
        SERVER_GPU_ID=0\n\
        TASK=\"{task_name}\"\n\
        DATA_IDX_FILE=\"journal/{dataset}/{dataset_type}/{N}client\"\n\n\
        cd easyFL\n\n\
        "

        # task_name = f"{dataset}_{dataset_type}_N{N}_K{K}_E{E}"
        # command = formated_command.format(
        #     task_name, algo, model, 1000, E, batch_size, K/N, task_name, dataset_type, N
        # )
            
        body_text = "python main.py  --task ${TASK}  --model ${MODEL}  --algorithm ${ALG} --sthr ${STHR} --wandb ${WANDB} --data_folder ${DATA_DIR}  --log_folder ${LOG_DIR}   --dataidx_filename ${DATA_IDX_FILE}   --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU}  --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} "

        dir_path = f"./run3/{dataset}/{dataset_type}/"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        file = open(dir_path + f"{task_name}_{algo}.sh", "w")
        file.write(header_text + command + body_text)
        file.close()