#!/bin/bash
#SBATCH -J default_ThermoMPNN
#SBATCH -t 48:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --qos=gpu_access

module load gcc/11.2.0
module load cuda/11.8

cd /proj/kuhl_lab/users/dieckhau/ThermoMPNN

/nas/longleaf/home/dieckhau/miniconda3/envs/proteinMPNN/bin/python train_thermompnn.py DEFAULT.yaml @
