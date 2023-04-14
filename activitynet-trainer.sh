#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=../anet_output/UPDATED_lr0.0001_bs16_stride_2_do0.out

module load anaconda
srun python trainer.py --annot_dir "../anet_annotations/activitynet-200-category.json" --num_classes 200 --epochs 50 --lr 0.0001 --dataset "Anet" --batch_size 16 --frames_per_clip 32 --pr 2 --tt_split 0.9 --str 2 --dropout 0
