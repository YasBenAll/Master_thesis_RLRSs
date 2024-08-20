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
from agents.state_encoders import GRUStateEncoder

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
    ),
    parser.add_argument(
        "--agent",
        type=str,
        required = False,
        default="MOPPO",
        choices=["MOSAC", "MOPPO"],
        help="Type of agent",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1e6,
        help="Total number of timesteps for training.",
    )
    return parser


import numpy as np
from agents.morl.evaluation import eval_mo
from agents.pgmorl import PGMORL, make_env




if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()
    decoder = torch.load(args.data_dir+"GeMS/decoder/"+args.exp_name+"/e5938782b93eca33d86f340ac2f09eb5b79aae6379d99a20964d768861abed32.pt", map_location=torch.device('cpu')).to(args.device)
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    env_id = "SlateTopK-BoredInf-v0-num_item1000-slate_size10"
    algo = PGMORL(
        env_id=env_id,
        num_envs=2,
        pop_size=4,
        warmup_iterations=5,
        evolutionary_iterations=20,
        num_weight_candidates=7,
        origin=np.array([0.0, 0.0]),
        args=args,
        decoder=decoder,
        buffer=RolloutBuffer,
        log=args.log,
        gamma = 0.8,
        device = device, 
        agent = args.agent
    )
    print("Training PGMORL")
    eval_env = make_env(env_id=env_id, seed=42, run_name="Sardine_pgmorl", gamma=0.8, observable=False, decoder=decoder, observation_shape = 16, args = args)()
    num_users_generated = 0
    algo.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=np.array([0.0, 0.0]),
        known_pareto_front=None,
        num_users_generated = num_users_generated
    )
    
    # Evaluation
    print("Evaluating PGMORL")
    
    env = make_env(env_id=env_id, seed=42, run_name="Sardine_pgmorl", gamma=0.8, observable=False, decoder=decoder, observation_shape = 16, args = args)
    mo_sync_env = mo_gym.MOSyncVectorEnv([env])
    StateEncoder = GRUStateEncoder
    state_encoder = StateEncoder(mo_sync_env, args)

    # Execution of trained policies
    for a in algo.archive.individuals:
        scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
            agent=a, env=env(), render=False, w = np.array([.5, .5]),state_encoder=state_encoder, num_envs=1, foo=True
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")

    print("Training and evaluating PGMORL done.")