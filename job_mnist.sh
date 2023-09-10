#!/bin/bash
#
#SBATCH --job-name=Journal-CADIS-FEDRL
#SBATCH --output=/vinserver_user/hung.nn184118/workspace/journal/fedtask/job-mnist.txt
#
#SBATCH --ntasks=1 --cpus-per-task=4 --gpus=1
#
sbcast -f /vinserver_user/hung.nn184118/workspace/journal/mnist.sh mnist.sh
srun sh mnist.sh