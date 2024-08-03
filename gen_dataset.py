import argparse
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import os
import pickle
import torch
from pathlib import Path
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle, EpsilonGreedyAntiOracle, RandomPolicy
from collections import OrderedDict
np.set_printoptions(precision=3)

seed = 2023
env_id = "ml-100k-v0"
data_dir = "C:/Users/Yassi/Documents/GitHub/Master_thesis_RLRSs/data/datasets/"
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir + "embeddings/").mkdir(parents=True, exist_ok=True)
lp = "oracle"
eps = 0.5
n_users = 100000

## Let's create the environment of our choice
env = mo_gym.make(env_id)
## If you want to work with Fully observable state, add a wrapper to the environment
env = IdealState(env)

## Generate a dataset of 10 users with 50% random actions and 50% greedy actions
if lp == "oracle":
    logging_policy = EpsilonGreedyOracle(epsilon = eps, env = env, seed = seed)
elif lp == "antioracle":
    logging_policy = EpsilonGreedyAntiOracle(epsilon = eps, env = env, seed = seed)
dataset = env.generate_dataset(n_users = n_users, policy = logging_policy, seed = seed, dataset_type="sb3_replay")

path = env_id + "_" + lp + "_epsilon" + str(eps) + "_seed" + str(seed) + "_n_users"+ str(n_users) + ".pt"
torch.save(dataset, data_dir + path)
torch.save(env.unwrapped.item_embedd, data_dir + "embeddings/" + path)
print("Dataset saved at: ", data_dir + path)
print("Embeddings saved at: ", data_dir + "embeddings/" + path)