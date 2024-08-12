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

from gems import gems
import pytorch_lightning as pl
from utils.parser import get_generic_parser
from utils.file import hash_config
from distutils.util import strtobool
from pathlib import Path
np.set_printoptions(precision=3)

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    args, _ = parser.parse_known_args(args)
    return parser

num_items = [100, 200, 300, 500, 1000, 10000]
slate_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 20]

# generate dataset
# for num_item in num_items:
#     for slate_size in slate_sizes:
#         print(f"num_item: {num_item}, slate_size: {slate_size}")
#         seed = 2023
#         env_id = f"SlateTopK-BoredInf-v0-num_item{num_item}-slate_size{slate_size}"
#         data_dir = "C:/Users/Yassi/Documents/GitHub/Master_thesis_RLRSs/data/datasets/"
#         Path(data_dir).mkdir(parents=True, exist_ok=True)
#         Path(data_dir + "embeddings/").mkdir(parents=True, exist_ok=True)
#         lp = "oracle"
#         eps = 0.5
#         n_users = 100000

#         ## Let's create the environment of our choice
#         env = mo_gym.make(env_id)
#         ## If you want to work with Fully observable state, add a wrapper to the environment
#         env = IdealState(env)

#         ## Generate a dataset of 10 users with 50% random actions and 50% greedy actions
#         if lp == "oracle":
#             logging_policy = EpsilonGreedyOracle(epsilon = eps, env = env, seed = seed)
#         elif lp == "antioracle":
#             logging_policy = EpsilonGreedyAntiOracle(epsilon = eps, env = env, seed = seed)
#         dataset = env.generate_dataset(n_users = n_users, policy = logging_policy, seed = seed, dataset_type="sb3_replay")

#         path = env_id + "_" + lp + "_epsilon" + str(eps) + "_seed" + str(seed) + "_n_users"+ str(n_users) + f"seed{seed}" +".pt"
#         torch.save(dataset, data_dir + path)
#         torch.save(env.unwrapped.item_embedd, data_dir + "embeddings/" + path)
#         print("Dataset saved at: ", data_dir + path)
#         print("Embeddings saved at: ", data_dir + "embeddings/" + path)

# pretrain GeMS


for num_item in num_items:
    for slate_size in slate_sizes:
        print(f"num_item: {num_item}, slate_size: {slate_size}")
        parser = get_parser(parents = [get_generic_parser(), gems.get_parser()])
        args = parser.parse_args()
        print(args)
        args.dataset=f"SlateTopK-BoredInf-v0-num_item{num_item}-slate_size{slate_size}_oracle_epsilon0.5_seed2023_n_users100000seed2023.pt"
        decoder = None

        pl.seed_everything(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        torch.set_float32_matmul_precision('medium')
        device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        if device.type != "cpu":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        print("### Pretraining GeMS ###")
        decoder_dir = args.data_dir + "GeMS/decoder/" + args.exp_name + "/"
        config_hash = hash_config(args, index=True)

        decoder = gems.train(args, config_hash) 