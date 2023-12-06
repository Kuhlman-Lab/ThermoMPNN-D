#!/bin/bash
#SBATCH -J sca8ProteinMPNN_020_0
#SBATCH -t 3-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --qos=gpu_access

module load gcc/11.2.0
module load cuda/11.8

cd /proj/kuhl_lab/users/dieckhau/ThermoMPNN/proteinmpnn/ 

/nas/longleaf/home/dieckhau/miniconda3/envs/proteinMPNN/bin/python training.py \
	--path_for_training_data /proj/kuhl_lab/datasets/pdb_2021aug02 \
	--path_for_outputs sca8ProteinMPNN_020_0 \
	--backbone_noise 0.2 \
	--side_chains 8

