#!/bin/bash
#SBATCH -J inference
#SBATCH -t 4:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc
# conda activate thermoMPNN
conda activate proteinMPNN

module load gcc
module load cuda

# fill in your repo location for ThermoMPNN
# repo_location="~/ThermoMPNN/"
repo_location='/proj/kuhl_lab/users/dieckhau/ThermoMPNN'

cd "$repo_location/analysis"

python custom_inference.py --pdb ../examples/pdbs/4ajy.pdb --chain B --model_path ../models/thermoMPNN_default.pt
