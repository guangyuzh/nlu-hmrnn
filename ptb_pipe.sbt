#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p40_4
#SBATCH --mem=100GB
#SBATCH --job-name=PTB_pipe
#SBATCH --mail-type=END
#SBATCH --mail-user=gz612@nyu.edu
#SBATCH --output=logs/ptb_pipe.out

module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

cd hierarchical-rnn
python3 char_class.py > logs/char_class.log
python3 -u ptb_test.py > logs/ptb_test.log

cd ../treebank
python3 evaluate.py > logs/ptb_eval.log

