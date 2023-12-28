#!/bin/bash
#SBATCH --job-name=CADIS-FEDRL-Journal
#SBATCH --output=/vinserver_user/hapq/journal/fedtask/job_proposal2.txt
#SBATCH --ntasks=1 --cpus-per-task=4 --gpus=1

# Your job commands go here
sbcast -f /vinserver_user/hapq/easyFL/proposal2.sh proposal2.sh
srun sh proposal2.sh