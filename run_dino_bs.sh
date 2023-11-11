#!/bin/bash

#SBATCH --job-name=dino_bs
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu # Partition (queue) name, stampede, biggpu, bigbatch

source /home-mscluster/${USER}/.bashrc
conda activate tb

# Run the script
# for pretraining parse in the pretrain {augmentation type}
# for training parse in train {augmentation type}
# for evaluation parse in evaluate {augmentation type}
python3 bs_dino.py pretrain default # pretraining with default augmentations