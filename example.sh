#!/bin/bash
# SBATCH --job-name=test
# SBATCH --time=72:00:00
# SBATCH -N 1
# SBATCH --ntasks-per-node=1
# SBATCH --partition=defq
# SBATCH --gres=gpu:1

# Load GPU drivers
 module load cuda11.3/toolkit/11.3.1

# Base directory for the experiment
mkdir $HOME/experiments
cd $HOME/experiments

python3.10 -m venv <name
source venv/bin/activate
# bin/pip install -r requirements.txt


# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment.
python /home/yal700/git/Master_thesis_RLRSs/test.py
deactivate