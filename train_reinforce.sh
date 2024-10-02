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

# Create a virtual environment in /var/scratch/$USER/
python3.10 -m venv /var/scratch/$USER/venv
source /var/scratch/$USER/venv/bin/activate
python -V

# Uncomment to install required packages
# python -m pip install --upgrade pip
# python -m pip install -r /home/yal700/git/Master_thesis_RLRSs/requirements.txt

# Define betas array and select based on SLURM_ARRAY_TASK_ID
betas=(2705 3751 4685 3688 6383)
SEED=${betas[SLURM_ARRAY_TASK_ID]}

echo "Beta (KL): $BETA"
echo "Lambda (click): $LAMBDA"
echo "Slate size: $SLATE_SIZE"
echo "Num items: $NUM_ITEMS"
# mkdir pretrain_gems


# Run the actual experiment
python /var/scratch/yal700/Master_thesis_RLRSs/train.py --agent reinforce --total-timesteps 5000 --val-interval 1000 --policy-frequency 1 --buffer-size 100 --env-id sardine/SlateTopK-Bored-v0 --morl False  --device cuda --observable True --slate-size 3  --exp-name final_actual --seed $SEED --env-id sardine/SlateTopK-Bored-v0 --num-items $NUM_ITEMS --env-embedds item_embeddings_numitems$SLATE_SIZE.npy --train True --test True
deactivate
