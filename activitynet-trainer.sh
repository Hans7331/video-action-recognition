#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/e50-lr-0.0001-all-pretrain-fixed_test.out

module load anaconda
srun python trainer.py --annot_dir "../anet_annotations/activitynet-3-category.json" --num_classes 3 --epochs 1 --lr 0.1 --dataset "Anet" --batch_size 32 --frames_per_clip 16 --pr 2
