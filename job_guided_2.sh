#!/bin/bash
#
#SBATCH --job-name=Journal-CADIS-FEDRL
#SBATCH --output=/vinserver_user/hung.nn184118/workspace/journal/fedtask/job-guided-2.txt
#
#SBATCH --ntasks=1 --cpus-per-task=4 --gpus=1
#
sbcast -f /vinserver_user/hung.nn184118/workspace/journal/guided-2.sh guided-2.sh
srun sh guided-2.sh