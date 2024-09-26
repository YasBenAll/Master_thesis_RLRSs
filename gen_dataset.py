import argparse
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import re
import time
import torch
from collections import OrderedDict
from distutils.util import strtobool
from pathlib import Path
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle, EpsilonGreedyAntiOracle, RandomPolicy
from utils.parser import get_generic_parser
from utils.file import hash_config

np.set_printoptions(precision=3)

def get_parser(parents=[], args=None):
    parser = argparse.ArgumentParser(parents=parents, add_help=False)
    parser.add_argument(
        "--dataset-multi",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Path to dataset for multi training",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=100000,
        help="Number of users to generate data for",
    )
    parser.add_argument(
        "--ideal-state",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use ideal state",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="oracle",
        help="Logging policy",
        choices=["oracle", "antioracle"],
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Epsilon for logging policy",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="sardine/SlateTopK-Bored-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--loading-bar",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use ideal state",
    )
    parser.add_argument(
        "--env-embedds",
        type=str,
        default="item_embeddings_numitems",
    )
    args, _ = parser.parse_known_args(args)
    return parser

def create_environment(args, env_id, slate_size, num_item):
    env = mo_gym.make(env_id, morl=False, slate_size=slate_size, num_items = num_item, env_embedds=f"{args.env_embedds}{num_item}.npy")
    if args.ideal_state:
        env = IdealState(env)
    return env

def select_logging_policy(args, env):
    if args.lp == "oracle":
        return EpsilonGreedyOracle(epsilon=args.eps, env=env, seed=args.seed)
    elif args.lp == "antioracle":
        return EpsilonGreedyAntiOracle(epsilon=args.eps, env=env, seed=args.seed)
    else:
        raise ValueError(f"Unknown logging policy: {args.lp}")

def save_dataset_and_embeddings(args, dataset, env, path_name):
    path_dataset = os.path.join(args.data_dir,'datasets', path_name+".pt")
    path_embeddings = os.path.join(args.data_dir,'datasets','embeddings', path_name+".npy")
    
    torch.save(dataset, os.path.join(path_dataset))
    np.save(os.path.join(path_embeddings), env.unwrapped.item_embedd)
    print(f"Dataset saved at: {path_dataset}")
    print(f"Embeddings saved at: {path_embeddings}")

def generate_dataset(args):
    env_id_str = re.sub(r'[\W_]+', '', args.env_id)
    path_name = f"{env_id_str}_{args.lp}_epsilon{args.eps}_seed{args.seed}_n_users{args.n_users}"

    env = create_environment(args, args.env_id)
    logging_policy = select_logging_policy(args, env)
    
    dataset = env.generate_dataset(
        n_users=args.n_users, 
        policy=logging_policy, 
        seed=args.seed, 
        dataset_type="sb3_replay",
        loading_bar = args.loading_bar
    )

    save_dataset_and_embeddings(args, dataset, env, path_name)

def generate_datasets_for_multiple_configs(args):
    num_items = [100, 500, 1000]
    slate_sizes = [3, 5, 10, 20]
    
    for num_item in num_items:
        for slate_size in slate_sizes:
            print(f"num_item: {num_item}, slate_size: {slate_size}")
            
            env_id = f"sardine/SlateTopK-Bored-v0"
            env_name = f"SlateTopK-Bored-v0-num_item{num_item}-slate_size{slate_size}"
            env = create_environment(args, env_id, slate_size=slate_size, num_item = num_item)
            logging_policy = select_logging_policy(args, env)

            dataset = env.generate_dataset(
                n_users=args.n_users, 
                policy=logging_policy, 
                seed=args.seed, 
                dataset_type="sb3_replay",
                loading_bar=args.loading_bar
            )

            env_id_str = re.sub(r'[\W_]+', '', env_name)
            path_name = f"{env_id_str}_{args.lp}_epsilon{args.eps}_seed{args.seed}_n_users{args.n_users}"
            
            save_dataset_and_embeddings(args, dataset, env, path_name)

def main():
    print("### Generating Dataset ###")
    parser = get_parser(parents=[get_generic_parser()])
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_dir, "production","datasets","embeddings")).mkdir(parents=True, exist_ok=True)

    if args.dataset_multi:
        generate_datasets_for_multiple_configs(args)
    else:
        generate_dataset(args)

if __name__ == "__main__":
    main()
