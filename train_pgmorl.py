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
    return parser


import numpy as np
from agents.morl.evaluation import eval_mo
from agents.pgmorl import PGMORL, make_env




if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()
    if args.observable:
        args.state_dim = 30

    decoder = torch.load(os.path.join(args.data_dir,"GeMS", "decoder", args.exp_name, args.decoder_name), map_location=torch.device('cpu')).to(args.device)
    pl.seed_everything(args.seed)

    
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
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
    )


    if args.train:

        print(f"Training PGMORL using {args.agent} on {args.env_id} with {args.total_timesteps} timesteps.")
        eval_env = make_env(env_id=args.env_id, seed=42, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)()
        num_users_generated = 0
        import time
        start = time.time()
        algo.train(
                total_timesteps=args.total_timesteps,
                eval_env=eval_env,
                ref_point=np.array([0.0, 0.0]),
                known_pareto_front=None,
                num_users_generated = num_users_generated
            )
        print("Training time: ", round((time.time() - start) / 60, 2), " minutes")

        if args.save:
            archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}.pkl")
            evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations.pkl")

            algo.save_pareto_archive(archive_filename, evaluations_filename)
            print(f"Pareto archive saved to {archive_filename}")



        env = make_env(env_id=args.env_id, seed=args.seed+2, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)
        mo_sync_env = mo_gym.MOSyncVectorEnv([env])
        if not args.observable:
            StateEncoder = GRUStateEncoder
            state_encoder = StateEncoder(mo_sync_env, args)
        else:
            state_encoder = None
        # Execution of trained policies multiple times 
        for _ in range(5):
            for a in algo.archive.individuals:
                scalarized, discounted_scalarized, reward, discounted_reward, info = eval_mo(
                    agent=a, env=env(), render=False, w = a.np_weights, state_encoder=state_encoder, num_envs=1, foo=True, seed=args.seed+_, observable=args.observable
                )
                print(f"Agent #{a.id}")
                print(f"Scalarized: {scalarized}")
                print(f"Discounted scalarized: {discounted_scalarized}")
                print(f"Vectorial: {reward}")
                print(f"Discounted vectorial: {discounted_reward}")
                print(f"Weights:{a.np_weights}")
                print(f"Info: {info['catalog_coverage']}")

    if args.test:
        # load ar
        archive_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}.pkl")
        evaluations_filename = re.sub(r"[^a-zA-Z0-9]+", '-', f"pareto_archive_{args.agent}_timesteps{int(args.total_timesteps)}_ranker{args.ranker}_env{args.env_id}_evaluations.pkl")

        algo.load_pareto_archive(archive_filename, evaluation_filename=evaluations_filename)

        env = make_env(env_id=args.env_id, seed=args.seed+2, run_name="Sardine_pgmorl", gamma=0.8, observable=args.observable, decoder=decoder, observation_shape = 16, args = args)
        mo_sync_env = mo_gym.MOSyncVectorEnv([env])
        if not args.observable:
            StateEncoder = GRUStateEncoder
            state_encoder = StateEncoder(mo_sync_env, args)
        else:
            state_encoder = None

        # Execution of trained policies multiple times 
        for _ in range(100):
            for a in algo.archive.individuals:
                scalarized, discounted_scalarized, reward, discounted_reward, info = eval_mo(
                    agent=a, env=env(), render=False, w = a.np_weights, state_encoder=state_encoder, num_envs=1, foo=True, seed=args.seed+_, observable=args.observable
                )
                print(f"Agent #{a.id}")
                print(f"Info: {info['catalog_coverage']}")
                print(f"Cumulative Clicks: {reward[0]}")
                print(f"Average intra-list diversity: {reward[0]/100}")
                print(f"Weights:{a.np_weights}")
                print(f"Scalarized: {scalarized}")
                print(f"Info: {info['catalog_coverage']}")


        import pickle
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        # Step 1: Load the saved Pareto archive

        # Load individuals and evaluations from the saved files
        with open(os.path.join("data","morl",archive_filename), 'rb') as f:
            archive_data = pickle.load(f)

        with open(os.path.join("data","morl",evaluations_filename), 'rb') as f:
            evaluations = pickle.load(f)

        # Assuming evaluations is a list of objectives: accuracy and diversity
        evaluations = np.array(evaluations)  # Convert evaluations to numpy array for easier manipulation

        # Step 2: Apply k-means clustering to identify families of policies

        try:
            num_clusters = 3  # Assuming 3 families; you can adjust this number as needed
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(evaluations)
        except:
            try:
                num_clusters = 2
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(evaluations)
            except:
                num_clusters = 1
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(evaluations)

        print(f"Clusters found: {np.unique(labels)}")
        # Step 3: Plotting the Pareto front with objectives
        plt.figure(figsize=(8, 6))

        # Accuracy vs Diversity Plot
        colors = ['r', 'b', 'g']  # Colors for each family, adjust as needed
        for cluster in np.unique(labels):
            cluster_evaluations = evaluations[labels == cluster]
            plt.scatter(cluster_evaluations[:, 0], cluster_evaluations[:, 1]/100, color=colors[cluster], label=f'Family {cluster}')

        # Optional: Connect points to create an approximate Pareto front
        # Sort evaluations by accuracy and plot to approximate the Pareto front
        sorted_evaluations = evaluations[np.argsort(evaluations[:, 0])]
        plt.plot(sorted_evaluations[:, 0], sorted_evaluations[:, 1]/100, color='k', linestyle='--', linewidth=1, label='Pareto Front')

        plt.title('Pareto Front Visualization',fontsize=20, )
        plt.xlabel('Cumulative clicks',fontsize=20, )
        plt.ylabel('Average Intra-list diversity',fontsize=20, )
        plt.legend()

        # Step 4: Save the plot
        filename = f'pareto_front_{args.agent}_slatesize{args.slate_size}_numitems{args.num_items}_timesteps_{args.total_timesteps}.png'
        plt.savefig(os.path.join("plots","morl",filename))

        print(f"plot saved in plots/morl/{filename}")