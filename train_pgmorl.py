import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import random
import re
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
from sardine.buffer.buffers import RolloutBuffer
from agents.state_encoders import GRUStateEncoder

import datetime

import numpy as np
import matplotlib.pyplot as plt



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
        choices=["mosac", "moppo"],
        help="Type of agent",
    )
    parser.add_argument(
        "--total-timesteps",
        type=float,
        default=100000.0,
        help="Total number of timesteps for training.",
    )
    parser.add_argument(
        "--steps-per-iteration",
        type=int,
        default=1e4,
        help="Total number of timesteps for training.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="sardine/SlateTopK-Bored-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--save",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Save the model",
    )
    parser.add_argument(
        "--train",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Train the model",
    )
    parser.add_argument(
        "--test",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Test the model",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of environments",
    )
    parser.add_argument(
        "--observable",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Test the model"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="gems",
        choices=["topk", "gems"],
        help="Type of ranker for slate generation",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of neurons in hidden layers of all models.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=3,
        help="Population size for evolutionary algorithms.",
    )
    parser.add_argument(
        "--evolutionary-iterations",
        type=int,
        default=3,
        help="Number of evolutionary iterations.",
    )
    parser.add_argument(
        "--random",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help = "Random weight selection",
    )
    return parser


import numpy as np
from agents.morl.evaluation import eval_mo
from agents.pgmorl import PGMORL, make_env



def make_pareto_front_plot(evaluations, name):
    plt.figure(figsize=(8, 6))
    # Plot the Pareto Front
    plt.scatter(evaluations[:, 0], evaluations[:, 1], color='b', label='Policies')
    # add catalog coverage to the plot as label to the points
    for i, txt in enumerate(evaluations):
        plt.annotate(f'{txt[2]:.2f}', (txt[0], txt[1]))
    

    sorted_evaluations = evaluations[np.argsort(evaluations[:, 0])]
    plt.plot(sorted_evaluations[:, 0], sorted_evaluations[:, 1], color='k', linestyle='--', linewidth=1, label='Pareto Front')

    plt.title('Pareto Front Visualization',fontsize=20, )
    plt.xlabel('Cumulative clicks',fontsize=20, )
    plt.ylabel('Average Intra-list diversity',fontsize=20, )
    plt.legend()

    filename = f'pareto_front_{args.agent}_slatesize{args.slate_size}_numitems{args.num_items}_timesteps_{args.total_timesteps}_{name}.png'


    # Step 4: Save the plot
    if args.random:
        filename = f'pareto_front_random_{args.agent}_slatesize{args.slate_size}_numitems{args.num_items}_timesteps_{args.total_timesteps}_{name}.png'


    Path(os.path.join("plots","morl")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join("plots","morl",filename))

    print(f"plot saved in plots/morl/{filename}")

if __name__ == "__main__":

    args = get_parser([get_generic_parser()]).parse_args()
    if args.observable:
        if args.ml100k:
            args.state_dim = 57
        else:
            args.state_dim = 30
    if args.ml100k: 
        args.num_items = 1682
    csv_filename = f"pgmorl-{args.ranker}_slatesize{args.slate_size}_num_items{args.num_items}_seed{str(args.seed)}{datetime.datetime.now()}"
    csv_filename = re.sub(r"[^a-zA-Z0-9]+", '-', csv_filename)

    decoder = torch.load(os.path.join(args.data_dir,"GeMS", "decoder", args.exp_name, args.decoder_name), map_location=torch.device('cpu')).to(args.device)
    pl.seed_everything(args.seed)

    
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.train:
        algo = PGMORL(
            env_id=args.env_id,
            num_envs=args.num_envs,
            pop_size=args.pop_size,
            warmup_iterations=args.warmup_iterations,
            evolutionary_iterations=args.evolutionary_iterations,
            num_weight_candidates=7,
            origin=np.array([0.0, 0.0]),
            args=args,
            decoder=decoder,
            buffer=RolloutBuffer,
            log=args.log,
            gamma = 0.8,
            device = device, 
            agent = args.agent,
            ranker = args.ranker,
            observable=args.observable,
            steps_per_iteration=args.steps_per_iteration,
            filename = csv_filename,
            ml100k = args.ml100k
        )

        print(f"Training PGMORL using {args.agent} on {args.env_id} with {args.total_timesteps} timesteps. random = {args.random}, observable = {args.observable}, ranker = {args.ranker}, decoder = {args.decoder_name}")
        Path(os.path.join("logs","morl")).mkdir(parents=True, exist_ok=True)
        with open(os.path.join("logs", "morl", f"{csv_filename}.log"), "w") as f:
            f.write(f"Training PGMORL using {args.agent} on {args.env_id} with {args.total_timesteps} timesteps. random = {args.random}, observable = {args.observable}, ranker = {args.ranker}, decoder = {args.decoder_name}\n")
            f.write(f"{args}")
        eval_env = make_env(env_id=args.env_id, seed=args.seed+1, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)()
        num_users_generated = 0
        import time
        start = time.time()
        algo.train(
                total_timesteps=args.total_timesteps,
                eval_env=eval_env,
                ref_point=np.array([0.0, 0.0]),
                known_pareto_front=None,
                num_users_generated = num_users_generated,
                csv_filename = csv_filename + "_train"
            )
        print("Training time: ", round((time.time() - start) / 60, 2), " minutes")
        print(os.path.join("logs","morl",f'{csv_filename}.log'))
        with open(os.path.join("logs","morl",f'{csv_filename}.log'), "a") as f:
            f.write(f"\nTraining time: {round((time.time() - start) / 60, 2)} minutes")
            f.write(f"\n Pareto front evaluations: {algo.archive.evaluations}")
            f.write(f"\n Catalog Coverages: {algo.archive.catalog_coverage}")
        if args.save:
            archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}")
            evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations")

            if args.random:
                archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_random_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}")
                evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_random_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations")


            algo.save_pareto_archive(archive_filename, evaluations_filename)
            print(f"Pareto archive saved to {archive_filename}")

        result = np.array([np.append(algo.archive.catalog_coverage, algo.archive.evaluations) for i, arr in enumerate(algo.archive.evaluations)])
        make_pareto_front_plot(result,"train")

        # env = make_env(env_id=args.env_id, seed=args.seed+2, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)
        # mo_sync_env = mo_gym.MOSyncVectorEnv([env])
        # if not args.observable:
        #     StateEncoder = GRUStateEncoder
        #     state_encoder = StateEncoder(mo_sync_env, args)
        # else:
        #     state_encoder = None
        # # Execution of trained policies multiple times 
        # for _ in range(5):
        #     for a in algo.archive.individuals:
        #         scalarized, discounted_scalarized, reward, discounted_reward, info = eval_mo(
        #             agent=a, env=env(), render=False, w = a.np_weights, state_encoder=state_encoder, num_envs=1, foo=True, seed=args.seed+_, observable=args.observable
        #         )
        #         print(f"Agent #{a.id}")
        #         print(f"Scalarized: {scalarized}")
        #         print(f"Discounted scalarized: {discounted_scalarized}")
        #         print(f"Vectorial: {reward}")
        #         print(f"Discounted vectorial: {discounted_reward}")
        #         print(f"Weights:{a.np_weights}")
        #         print(f"Info: {info['catalog_coverage']}")

    if args.test:
        algo = PGMORL(
            env_id=args.env_id,
            num_envs=args.num_envs,
            pop_size=args.pop_size,
            warmup_iterations=args.warmup_iterations,
            evolutionary_iterations=args.evolutionary_iterations,
            num_weight_candidates=7,
            origin=np.array([0.0, 0.0]),
            args=args,
            decoder=decoder,
            buffer=RolloutBuffer,
            log=args.log,
            gamma = 0.8,
            device = device, 
            agent = args.agent,
            ranker = args.ranker,
            observable=args.observable,
            steps_per_iteration=args.steps_per_iteration,
            filename = csv_filename,
            ml100k = args.ml100k,
        )

        # load ar
        archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}")
        evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations")

        if args.random:
            archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_random_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}")
            evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_random_{args.agent}_slatesize_{args.slate_size}_numitems{args.num_items}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations")


        algo.load_pareto_archive(archive_filename, evaluation_filename=evaluations_filename)
        eval_env = make_env(env_id=args.env_id, seed=args.seed+1, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)()
        test_env = make_env(env_id=args.env_id, seed=args.seed+2, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)
       
        if not args.observable:
            StateEncoder = GRUStateEncoder
            mo_sync_env = mo_gym.MOSyncVectorEnv([test_env])
            state_encoder = StateEncoder(mo_sync_env, args)
        else:
            state_encoder = None

        # Execution of trained policies multiple times 
        agent_dict = {}

        algo._PGMORL__eval_all_agents(eval_env=test_env(),
            evaluations_before_train=algo.archive.evaluations,
            ref_point=np.array([0.0, 0.0]),
            known_pareto_front=None,
            name = 'test',
            num_episodes=500
            )
        # print(evaluations)
        result = np.array([np.append( algo.archive.evaluations,algo.archive.catalog_coverage) for i, arr in enumerate(algo.archive.evaluations)])

        make_pareto_front_plot(result,"test")


