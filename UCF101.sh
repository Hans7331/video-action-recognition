#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/all-pretrain-e50-lr-0.0001.out

module load anaconda
srun python trainer.py --annot_dir "UCF101/trainlist01.txt" --num_classes 101 --epochs 50 --lr 0.0001 --dataset "UCF101" --batch_size 8 --frames_per_clip 16 --pr 2
