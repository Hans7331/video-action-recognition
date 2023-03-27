#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --output=anet_output/activitynet_output_module_3_test.out

module load anaconda
srun python trainer.py --annot_dir "anet_annotations/activitynet-3-category.json" --num_classes 3 --epochs 3 --lr 0.1 --dataset "Anet" --batch_size 32 --frames_per_clip 32 --pr 2
