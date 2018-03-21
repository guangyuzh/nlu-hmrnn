#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=Char_class
#SBATCH --mail-type=END
#SBATCH --mail-user=gz612@nyu.edu
#SBATCH --output=logs/char_class.out

module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

python3 -u char_class.py > logs/char_class.log
