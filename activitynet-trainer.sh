#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/e50-lr0.000001_do-0.1_tube-False.out

module load anaconda
srun python trainer.py --annot_dir "../anet_annotations/activitynet-200-category.json" --num_classes 200 --epochs 50 --lr 0.00001 --dataset "Anet" --batch_size 32 --frames_per_clip 32 --pr 2
