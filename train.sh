#!/bin/bash
#SBATCH --job-name=train13
#SBATCH --output=train13.out
#SBATCH --error=train13.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode074


# --- Environment setup ---
source ~/.bashrc
conda activate pyg

python train_span.py
