#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/test_resnet.out

module load anaconda
srun python trainer.py --annot_dir "../anet_annotations/activitynet-21-category.json" --num_classes 21 --epochs 1 --lr 0.01 --dataset "Anet" --batch_size 8 --frames_per_clip 32 --pr 2 --tt_split 0.9
