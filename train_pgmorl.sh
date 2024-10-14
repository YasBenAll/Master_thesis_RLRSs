#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -C gpunode
#SBATCH --gres=gpu:1
#SBATCH --array=0-4  # For indexing betas array

echo "Starting job $SLURM_JOB_ID"
echo "Slate size: $SLATE_SIZE"
echo "Num items: $NUM_ITEMS"
echo "ENV ID: $ENV_ID"
echo "Seed: $SEED"
echo "Ranker: $RANKER"
echo "Timesteps: $TIMESTEPS"
echo "Steps per iteration: $STEPS_PER_ITERATION"

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
seeds=(2705 3751 4685 3688 6383)
SEED=${seeds[SLURM_ARRAY_TASK_ID]}

# Run the actual experiment
python /var/scratch/yal700/Master_thesis_RLRSs/train_pgmorl.py \
    --exp-name final_actual \
    --env-embedds item_embeddings_numitems${NUM_ITEMS}.npy \
    --num-items $NUM_ITEMS \
    --slate-size ${SLATE_SIZE} \
    --total-timesteps $TIMESTEPS \
    --steps-per-iteration $STEPS_PER_ITERATION \
    --train True --test True \
    --log False --num-envs 4 \
    --observable True
    --agent mosac \
    --pop-size 5 \
    --evolutionary-iterations 4 \
    --warmup-iterations 5