#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --job-name=task_qa_peephole
#SBATCH --mail-type=END,BEGIN
#SBATCH --mail-user=mp4504@nyu.edu
#SBATCH --output=logs/qa_class_peephole.out

module purge
module load tensorflow/python3.6/1.5.0 cuda/9.0.176

python3 -u task_qa.py --dev > logs/dev_qa_peephole.log && \
python3 -u task_qa.py > logs/task_qa_peephole.log
