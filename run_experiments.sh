#!/bin/bash
# SBATCH --job-name=test
# SBATCH --time=00:01:00
# SBATCH -N 1
# SBATCH -C A4000
# SBATCH --ntasks-per-node=1
# SBATCH --partition=defq
# SBATCH --gres=gpu:1


echo "Starting job $SLURM_JOB_ID"

export CUDA_HOME=/cm/shared/apps/cuda/11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Load GPU drivers
. /etc/bashrc
. ~/.bashrc
module load cuda11.7/toolkit/11.7

./cuda-app opts

nvidia-smi

module avail cuda


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
python /home/yal700/git/Master_thesis_RLRSs/test.py
deactivate