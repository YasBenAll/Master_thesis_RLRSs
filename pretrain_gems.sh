#!/bin/sh
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:1
#SBATCH --array=0-4  # For indexing betas array

echo "Starting job $SLURM_JOB_ID"

# Load GPU drivers
. /etc/bashrc
. ~/.bashrc
module load cuda11.7/toolkit/11.7

nvidia-smi

# Create a virtual environment in /var/scratch/$USER/
python3.10 -m venv /var/scratch/$USER/venv
source /var/scratch/$USER/venv/bin/activate
python -V

# Uncomment to install required packages
# python -m pip install --upgrade pip
# python -m pip install -r /home/yal700/git/Master_thesis_RLRSs/requirements.txt

# Define betas array and select based on SLURM_ARRAY_TASK_ID
betas=(0.1 0.2 0.5 1.0 2.0)
beta=${betas[SLURM_ARRAY_TASK_ID]}

mkdir pretrain_gems


# Create a unique directory for each run
echo $$  # Print process ID
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment
python /var/scratch/yal700/Master_thesis_RLRSs/pretrain_gems.py --latent-dim 16 --multi True --slate-size 3 --num-items 100000 --latent-dim 16 --seed 2023 --n-users 10 --concurrent False --env-id SlateTopK-BoredInf-v0

deactivate
