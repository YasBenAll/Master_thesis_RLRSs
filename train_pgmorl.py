import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import random
import mo_gymnasium as mo_gym
import numpy as np
from agents import sac, pgmorl
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config
import argparse
import os
import subprocess
from distutils.util import strtobool
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers.record_video import RecordVideo
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.common.evaluation import seed_everything
from sardine.buffer.buffers import RolloutBuffer

def get_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--log",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Log results on wandb",
    ),
    parser.add_argument(
        "--state-dim",
        type=int,
        default=16,
        help="State dimension in POMDP settings.",
    ),
    parser.add_argument(
        "--sampled-seq-len",
        type=int,
        default=10,
        help="Number of timesteps to be sampled from replay buffer for each trajectory (only for POMDP)",
    ),
    parser.add_argument(
        "--ideal-se",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Ideal embeddings used in the state encoder",
    ),
    parser.add_argument(
        "--item-dim-se",
        type=int,
        default=16,
        help="Dimension of item embeddings in the state encoder.",
    ),
    parser.add_argument(
        "--click-dim-se",
        type=int,
        default=2,
        help="Dimension of click embeddings in the state encoder.",
    )
    parser.add_argument(
        "--num-layers-se",
        type=int,
        default=2,
        help="Number of layers in the state encoder.",
    )
    return parser


import numpy as np
from morl_baselines.common.evaluation import eval_mo
from agents.pgmorl import PGMORL, make_env




if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()
    decoder = torch.load(args.data_dir+"GeMS/decoder/"+args.exp_name+"/test-run.pt").to(args.device)
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    env_id = "ml-100k-v0"
    algo = PGMORL(
        env_id=env_id,
        num_envs=2,
        pop_size=6,
        warmup_iterations=1,
        evolutionary_iterations=1,
        num_weight_candidates=1,
        origin=np.array([0.0, 0.0]),
        args=args,
        decoder=decoder,
        buffer=RolloutBuffer,
        log=args.log,
        num_items = 1682,
        gamma = 0.8
    )
    print("Training PGMORL")
    eval_env = make_env(env_id=env_id, seed=42, run_name="Sardine_pgmorl", gamma=0.8, observable=False, decoder=decoder, observation_shape = 16, args = args)()
    algo.train(
        total_timesteps=int(1e5),
        eval_env=eval_env,
        ref_point=np.array([0.0, 0.0]),
        known_pareto_front=None,
    )
    env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()  # idx != 0 to avoid taking videos
    
    # Execution of trained policies
    for a in algo.archive.individuals:
        scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
            agent=a, env=env, w=np.array([1.0, 1.0]), render=False
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")
