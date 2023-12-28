#!/bin/bash
#SBATCH --job-name=CADIS-FEDRL-Journal
#SBATCH --output=/vinserver_user/hapq/journal/fedtask/job_scaffold.txt
#SBATCH --ntasks=1 --cpus-per-task=4 --gpus=1 

# Your job commands go here
sbcast -f /vinserver_user/hapq/easyFL/scaffold.sh scaffold.sh
srun sh scaffold.sh