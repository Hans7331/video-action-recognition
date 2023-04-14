#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/cl_e50-c200-bs16-do0_lr0.000001_test.out

module load anaconda
srun python trainer.py --annot_dir "../anet_annotations/activitynet-3-category.json" --num_classes 3 --epochs 1 --lr 0.000001 --dataset "Anet" --batch_size 16 --frames_per_clip 32 --pr 2 --tt_split 0.9
