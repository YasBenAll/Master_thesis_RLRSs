#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1
#SBATCH --array=0-4  # For indexing betas array

echo "$SLATESIZE $LD $SLURM_ARRAY_TASK_ID $LAMBDA $NUM_ITEMS $BETA"
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
BETA=${betas[SLURM_ARRAY_TASK_ID]}

echo "Beta (KL): $BETA"
echo "Lambda (click): $LAMBDA"
echo "Slate size: $SLATE_SIZE"
echo "Latent dim: $LD"
echo "Num items: $NUM_ITEMS"
echo "Number of users: $N_USERS"
# mkdir pretrain_gems


# # Create a unique directory for each run
# echo $$  # Print process ID
# mkdir o`echo $$`
# cd o`echo $$`

# Run the actual experiment
python /var/scratch/yal700/Master_thesis_RLRSs/pretrain_gems.py --lambda-click $LAMBDA --slate-size $SLATE_SIZE --exp-name final --latent-dim $LD --lambda-KL $BETA --num-items $NUM_ITEMS --n-users $N_USERS --device cuda --multi False  --seed 2023 --concurrent False --env-id SlateTopK-Bored-v0

deactivate
