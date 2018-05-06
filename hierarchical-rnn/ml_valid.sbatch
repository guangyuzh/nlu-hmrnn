#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4
#SBATCH --mem=16GB
#SBATCH --job-name=Char_class
#SBATCH --mail-type=END,BEGIN
#SBATCH --mail-user=mp4504@nyu.edu
#SBATCH --output=logs/char_class.out


# configuration type to select in config.yml
# if not defined, will use the default configuration
CONFIG='quick_nets'

module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

python3 -u ml_valid.py --config $CONFIG > logs/ml_with_valid.log
