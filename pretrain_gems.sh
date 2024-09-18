#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:1


echo "Starting job $SLURM_JOB_ID"

# Load GPU drivers
. /etc/bashrc
. ~/.bashrc
module load cuda11.7/toolkit/11.7

nvidia-smi

# create a virtual environment in /var/scratch/$USER/
python3.10 -m venv /var/scratch/$USER/venv
source /var/scratch/$USER/venv/bin/activate
python -V

# Install the required packages
# python -m pip install --upgrade pip
# python -m pip install -r /home/yal700/git/Master_thesis_RLRSs/requirements.txt

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment.
python /var/scratch/yal700/Master_thesis_RLRSs/pretrain_gems.py
deactivate