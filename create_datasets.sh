#!/bin/bash

#SBATCH --job-name=datasets
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu # Partition (queue) name, stampede, biggpu, bigbatch

source /home-mscluster/${USER}/.bashrc
conda activate tb_gpu

# Run the training script
python3 create_bone_supp.py