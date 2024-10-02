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
python /var/scratch/yal700/Master_thesis_RLRSs/train.py --agent sac --total-timesteps 500000 --val-interval 1000  --buffer-size 100 --env-id sardine/SlateTopK-Bored-v0 --morl False --ranker topk --decoder-name SlateTopKBoredv0numitem${NUM_ITEMS}slatesize${SLATE_SIZE}_oracle_epsilon0.5_seed2023_n_users100000.ptkl_divergence0.1_lambda_click1.0_latentdim16 --device cuda --slate-size $SLATE_SIZE --singleq true --exp-name final_actual --seed $SEED --num-items $NUM_ITEMS --autotune True --observable True --learning-starts 10000 --n-val-episodes 25 --tau 0.05 --batch-size 32 --env-embedds item_embeddings_numitems$NUM_ITEMS.npy --train True --test True  
