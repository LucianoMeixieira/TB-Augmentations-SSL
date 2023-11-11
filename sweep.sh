#!/bin/bash

#SBATCH --job-name=sweep
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu # Partition (queue) name, stampede, biggpu, bigbatch

source /home-mscluster/${USER}/.bashrc
conda activate tb

# Run the script
python3 sweep.py dino bone_supp linear 