#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --job-name=sinusoidal
#SBATCH --mail-type=END
#SBATCH --mail-user=mp4504@nyu.edu
#SBATCH --output=logs/sinusoidal.out

module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

python3 -u sinusoidal.py > logs/sinusoidal.log

