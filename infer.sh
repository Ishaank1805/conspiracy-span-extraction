#!/bin/bash
#SBATCH --job-name=infer13
#SBATCH --output=infer13.out
#SBATCH --error=infer13.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode077


# --- Environment setup ---
source ~/.bashrc
conda activate pyg

python infer_span.py
