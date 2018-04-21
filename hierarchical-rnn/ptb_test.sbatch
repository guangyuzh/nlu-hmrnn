#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4
#SBATCH --mem=100GB
#SBATCH --job-name=PTB_test
#SBATCH --mail-type=END
#SBATCH --mail-user=gz612@nyu.edu
#SBATCH --output=logs/ptb_test.out

module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

python3 -u ptb_test.py > logs/ptb_test.log
