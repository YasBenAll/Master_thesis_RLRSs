import argparse
import datetime
import os
import csv
import random
import time
from distutils.util import strtobool
import sardine
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# import scipy.optimize as sco

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path

from .buffer import DictReplayBuffer, POMDPDictReplayBuffer
from .wrappers import IdealState
from .state_encoders import GRUStateEncoder, TransformerStateEncoder

from utils.parser import get_generic_parser
from utils.file import hash_config, args2str
from .utils.memory_usage import log_memory_usage


torch.set_float32_matmul_precision('high')

DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    # Training arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="SlateBorNoInf-ucar-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--observable",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, an observation with full state environment will be passed.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=50000,
        help="Number of timesteps between validation episodes.",
    )
    parser.add_argument(
        "--n-val-episodes",
        type=int,
        default=10,
        help="Number of validation episodes.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(1e6),
        help="the replay memory buffer size",
    )
    parser.add_argument(
        "--sampled-seq-len",
        type=int,
        default=100,
        help="Number of timesteps to be sampled from replay buffer for each trajectory (only for POMDP)",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=1e4, help="timestep to start learning"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="the batch size of sample from the reply memory",
    )
    # SAC arguments
    parser.add_argument(
        "--gamma", type=float, default=0.8, help="the discount factor gamma"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="target smoothing coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        default=0.1,
        help="the scale of exploration noise",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the policy network optimizer",
    )
    parser.add_argument(
        "--q-lr",
        type=float,
        default=1e-3,
        help="the learning rate of the Q network network optimizer",
    )
    parser.add_argument(
        "--behavior-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the behavior loss optimizer",
    )
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=2,
        help="the frequency of training policy update for actor loss",
    )
    parser.add_argument(
        "--hyper-frequency",
        type=int,
        default=2,
        help="the frequency of training policy update for hyper-actor loss.",
    )
    parser.add_argument(
        "--behavior-frequency",
        type=int,
        default=2,
        help="the frequency of training policy update for behavior loss.",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=1,
        help="the frequency of updates for the target networks",
    )
    parser.add_argument(
        "--target-actor-frequency",
        type=int,
        default=1,
        help="the frequency of updates for the target actor",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization",
    )
    parser.add_argument(
        "--n-updates", type=int, default=1, help="Number of Q updates per sample."
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of neurons in hidden layers of all models.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="State dimension in POMDP settings.",
    )
    parser.add_argument(
        "--reparam-std",
        type=float,
        default=0.1,
        help="Standard deviation for the reparameterization trick.",
    )
    parser.add_argument(
        "--hyper-weight",
        type=float,
        default=0.1,
        help="Weight for the hyper-actor loss.",
    )
    parser.add_argument(
        "--latent-high",
        type=float,
        default=1.0,
        help="Higher bound on the latent space.",
    )
    parser.add_argument(
        "--latent-low",
        type=float,
        default=-1.0,
        help="Lower bound on the latent space.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Dimension of the latent space.",
    )
    parser.add_argument(
        "--raw-features",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Use raw item features as item embeddings instead of projecting them.",
    )
    parser.add_argument(
        "--state-encoder",
        type=str,
        default="gru",
        choices=["gru", "transformer"],
        help="Type of state encoder (only for POMDP)",
    )
    parser.add_argument(
        "--ideal-se",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Ideal embeddings used in the state encoder",
    )
    parser.add_argument(
        "--shared-se",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Shared state encoder across all actors and belief networks",
    )
    parser.add_argument(
        "--item-dim-se",
        type=int,
        default=16,
        help="Dimension of item embeddings in the state encoder.",
    )
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
    parser.add_argument(
        "--num-heads-se",
        type=int,
        default=4,
        help="Number of heads in the state encoder (only for Transformer).",
    )
    parser.add_argument(
        "--dropout-rate-se",
        type=float,
        default=0.1,
        help="Dropout rate in the state encoder (only for Transformer).",
    )
    parser.add_argument(
        "--forward-dim-se",
        type=int,
        default=64,
        help="Feed-forward net dimension in the state encoder (only for Transformer).",
    )
    return parser


def make_env(
    env_id,
    idx,
    observable,
    args,
):
    def thunk():
        env = gym.make(env_id, morl = args.morl, slate_size = args.slate_size, env_embedds=args.env_embedds, num_topics = args.num_topics, ml100k=args.ml100k)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if observable:
            env = IdealState(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_size, state_dim, latent_dim):
        super().__init__()

        if state_dim is None:
            state_dim = np.array(env.single_observation_space.shape).prod()

        self.model = nn.Sequential(
            nn.Linear(
                state_dim + latent_dim,
                hidden_size
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, torch.sqrt(torch.tensor(2)))
            m.bias.data.fill_(0.0)

    def forward(self, x, a):
        x = torch.cat([x, a], dim = -1)
        return self.model(x)


class Actor(nn.Module):
    def __init__(self, env, hidden_size, state_dim, slate_size, latent_dim, latent_high, latent_low, reparam_std, item_features, raw_features):
        super().__init__()

        if state_dim is None:
            state_dim = np.array(env.single_observation_space.shape).prod()
        self.slate_size = slate_size
        self.item_features = nn.Embedding.from_pretrained(torch.tensor(item_features), freeze=True).float() # Raw item representations
        self.reparam_std = reparam_std # Standard deviation for the reparameterization trick
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_dim)
        if raw_features: # Use raw item features as item embeddings
            self.item_map = lambda x: x
        else: # Project raw features to obtain item embeddings
            self.item_map = nn.Linear(item_features.shape[1], latent_dim)
        # action rescaling
        self.register_buffer(
            "latent_high",
            torch.tensor(latent_high),
        )
        self.register_buffer(
            "latent_low",
            torch.tensor(latent_low),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, torch.sqrt(torch.tensor(2)))
            m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def get_action(self, x):
        mean = self(x)
        eps = torch.randn_like(mean)
        x_t = mean + eps * self.reparam_std  # for reparameterization trick (mean + std * N(0,1))
        #hyper_action = x_t * self.action_scale + self.action_bias
        hyper_action = torch.clamp(x_t, min=self.latent_low, max=self.latent_high)

        return hyper_action

    def effect2hyper_map(self, effect_action):
        # Infer the estimated hyper-action Z^ from the effect-action (i.e. slate) A by mean pooling the slate item embeddings
        item_features = self.item_features(effect_action) # From item ID to item raw features: (B, K) -> (B, K, item_dim)
        item_embedd = self.item_map(item_features) # From item raw features to item embeddings: (B, K, item_dim) -> (B, K, action_dim)
        hyper_action = torch.mean(item_embedd, dim = 1) # From item embeddings to estimated hyper-action: (B, K, action_dim) -> (B, action_dim)

        return hyper_action

    def hyper2effect_map(self, hyper_action):
        # Infer the effect-action (i.e. slate) A from the hyper-action Z by calculating the top-k from dot product with item embeddings
        dot_product = hyper_action @ self.item_map(self.item_features.weight.float()).t() # (num_envs, num_items)
        dot_product = dot_product.cpu().numpy()
        ind = np.argpartition(dot_product, - self.slate_size)[:, - self.slate_size:] # (num_envs, slate_size)
        top_dot_product = np.take_along_axis(dot_product, ind, axis=-1) # (num_envs, slate_size)
        top_ind = np.flip(np.argsort(top_dot_product), axis=-1) # (num_envs, slate_size)
        effect_action = np.take_along_axis(ind, top_ind, axis=-1) # Slate

        return effect_action


def train(args):
    run_name = f"{args.env_id}__{args.run_name}__{args.seed}__{int(time.time())}"
    if args.track == "wandb":
        import wandb
    elif args.track == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )


    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_filename = f"sac_-seed{str(args.seed)}"

    import re
    # Using regex to find the numbers after 'numitem' and 'slatesize'
    numitem_match = re.search(r'numitem(\d+)', args.decoder_name)
    numitem_value = numitem_match.group(1) if numitem_match else None

    csv_filename2 = f"hac_slatesize{args.slate_size}_num_items{numitem_value}_seed{str(args.seed)}-{DATE}"
    # remove special characters
    csv_filename = re.sub(r"[^a-zA-Z0-9]+", '-', csv_filename)+".log"
    csv_filename2 = re.sub(r"[^a-zA-Z0-9]+", '-', csv_filename2)+".log"

    csv_path = "logs/" + csv_filename
    csv_path2 = "logs/" + csv_filename2
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path2, "w") as f:
        f.write(f"Start: {start}\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Config: {args}\n")
    csv_path = "logs/" + csv_filename + ".log"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # env setup
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args
            )
        ]
    )
    val_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args
            )
        ]
    )

    item_features = envs.envs[0].unwrapped.item_embedd
    slate_size = envs.envs[0].unwrapped.slate_size
    if args.raw_features: # The latent space is the item raw feature space
        args.latent_dim = item_features.shape[1]
    actor = Actor(
        envs, args.hidden_size, args.state_dim, slate_size, args.latent_dim,
        args.latent_high, args.latent_low, args.reparam_std, item_features,
        args.raw_features
    ).to(args.device)
    actor_target = Actor(
        envs, args.hidden_size, args.state_dim, slate_size, args.latent_dim,
        args.latent_high, args.latent_low, args.reparam_std, item_features,
        args.raw_features
    ).to(args.device)
    actor_target.load_state_dict(actor.state_dict())
    qf = SoftQNetwork(envs, args.hidden_size, args.state_dim, args.latent_dim).to(args.device)
    qf_target = SoftQNetwork(envs, args.hidden_size, args.state_dim, args.latent_dim).to(args.device)
    qf_target.load_state_dict(qf.state_dict())
    actor_params = list(actor.parameters())
    critic_params = list(qf.parameters())
    if not args.observable:
        if args.state_encoder == "gru":
            StateEncoder = GRUStateEncoder
        elif args.state_encoder == "transformer":
            StateEncoder = TransformerStateEncoder
        else:
            StateEncoder = None
        actor_state_encoder = StateEncoder(envs, args).to(args.device)
        actor_state_encoder_target = StateEncoder(envs, args).to(args.device)
        actor_state_encoder_target.load_state_dict(actor_state_encoder.state_dict())
        qf_state_encoder = actor_state_encoder if args.shared_se else StateEncoder(envs, args).to(args.device)
        qf_state_encoder_target = StateEncoder(envs, args).to(args.device)
        qf_state_encoder_target.load_state_dict(qf_state_encoder.state_dict())
        actor_params += list(actor_state_encoder.parameters())
        critic_params += list(qf_state_encoder.parameters())

    q_optimizer = optim.Adam(critic_params, lr=args.q_lr)
    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)
    behavior_optimizer = optim.Adam(actor_params, lr=args.behavior_lr)

    envs.single_observation_space.dtype = np.dtype(np.float32)
    click_space = spaces.MultiBinary(n = slate_size)
    hyper_action_space = spaces.Box(low = args.latent_low, high = args.latent_high, shape=(args.latent_dim,), dtype=np.float32)
    if args.observable:
        rb = DictReplayBuffer(
            args.buffer_size,
            spaces.Dict(
                {
                    "obs": envs.single_observation_space,
                    "clicks": click_space,
                    "slate": envs.single_action_space
                }
            ),
            hyper_action_space,
            args.device,
            handle_timeout_termination=True,
        )
    else:
        rb = POMDPDictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            hyper_action_space,
            args.sampled_seq_len,
            args.device,
            handle_timeout_termination=True,
        )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if not args.observable:
        actor_state_encoder.reset()
    hyper_action_space.seed(args.seed)

    max_val_return = 10e-5
    for global_step in range(args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            with torch.inference_mode():
                hyper_actions = np.array(
                    [hyper_action_space.sample() for _ in range(envs.num_envs)]
                )
                hyper_actions = torch.tensor(hyper_actions)
                effect_actions = actor.hyper2effect_map(hyper_actions)
        else:
            with torch.inference_mode():
                if args.observable:
                    obs_d = torch.Tensor(obs)
                else:
                    obs_d = actor_state_encoder.step(obs)
                hyper_actions = actor.get_action(obs_d)
                effect_actions = actor.hyper2effect_map(hyper_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(effect_actions)

        # Run validation episodes
        if global_step % args.val_interval == 0:
            val_start_time = time.time()
            val_obs, _ = val_envs.reset(seed=args.seed + 1)
            if not args.observable:
                actor_state_encoder.reset()
            ep = 0
            cum_boredom = 0
            val_returns, val_lengths, val_boredom, val_diversity, val_catalog_coverage = [], [], [], [], []
            val_errors, val_errors_norm = [], []
            ep_rewards, pred_q_values = [], []
            while ep < args.n_val_episodes:
                with torch.inference_mode():
                    if args.observable:
                        val_obs = torch.Tensor(val_obs)
                    else:
                        val_obs = actor_state_encoder.step(val_obs)
                    
                    val_hyper_action = actor.get_action(val_obs)
                    val_effect_action = actor.hyper2effect_map(val_hyper_action)
                    pred_q_values.append(qf(val_obs, val_hyper_action).item())
                (
                    val_next_obs,
                    val_rewards,
                    _,
                    _,
                    val_infos,
                ) = val_envs.step(val_effect_action)
                ep_rewards.append(val_rewards[0])
                val_obs = val_next_obs
                if "final_info" in val_infos:
                    if not args.observable:
                        actor_state_encoder.reset()
                    max_t = len(ep_rewards) - 20
                    if max_t > 0:
                        gamma_pow = np.power(args.gamma, np.arange(max_t))
                        discounted_returns = np.array(
                            [
                                (gamma_pow[:-i] * ep_rewards[i:max_t]).sum()
                                if i > 0
                            else (gamma_pow * ep_rewards[:max_t]).sum()
                                        for i in range(len(gamma_pow))
                            ]
                        )
                        val_errors.extend(pred_q_values[:max_t] - discounted_returns)
                        val_errors_norm.extend(np.abs(discounted_returns - pred_q_values[:max_t]) / discounted_returns.clip(min=0.01))
                    for info in val_infos["final_info"]:
                        # Skip the envs that are not done
                        if info is None:
                            continue
                        val_returns.append(info["episode"]["r"])
                        val_lengths.append(info["episode"]["l"])
                        val_diversity.append(info["diversity"])
                        val_boredom.append(cum_boredom)
                        cum_boredom = 0
                        ep += 1
                        ep_rewards, pred_q_values = [], []
                else:
                    cum_boredom += (1.0 if np.sum(val_infos["bored"][0] == True) > 0 else 0.0)

            if np.mean(val_returns) > max_val_return:
                max_val_return = np.mean(val_returns)
                # save the best model
                Path(os.path.join("data", "topk_hac")).mkdir(parents=True, exist_ok=True)
                torch.save(
                    actor.state_dict(),
                    os.path.join("data", "topk_hac", f"actor_best_slatesize{args.slate_size}_numitem{args.num_items}_{args.seed}.pt"),
                )
            with open(csv_path2, "a") as f:
                f.write(
                    f"\nStep {global_step}: clicks={np.mean(val_returns):.2f}, clicks_se={np.mean(val_returns)/np.sqrt(len(val_returns)):.2f}, diversity={np.mean(val_diversity):.2f}, diversity_se={np.mean(val_diversity)/np.sqrt(len(val_diversity))}, catalog coverage={np.mean(val_catalog_coverage):.2f}, catalog coverage_se={np.std(val_catalog_coverage)/np.sqrt(len(val_catalog_coverage)):.2f}"
                )
                log_memory_usage(file_path=csv_path2 ,step=global_step, tag=None)

            print(
                f"Step {global_step}: return={np.mean(val_returns):.2f} (+- {np.std(val_returns):.2f}), boredom={np.mean(val_boredom):.2f}"
            )
            if args.track == "wandb":
                wandb.log(
                    {
                        "val_charts/episodic_return": np.mean(val_returns),
                        "val_charts/episodic_length": np.mean(val_lengths),
                        "val_charts/q_error": np.mean(val_errors),
                        "val_charts/q_error_norm": np.mean(val_errors_norm),
                        "val_charts/SPS": int(np.sum(val_lengths) / (time.time() - val_start_time)),
                        "val_charts/boredom": np.mean(val_boredom),
                    },
                    global_step,
                )
            elif args.track == "tensorboard":
                writer.add_scalar(
                "val_charts/episodic_return", np.mean(val_returns), global_step
                )
                writer.add_scalar(
                    "val_charts/episodic_length", np.mean(val_lengths), global_step
                )
                writer.add_scalar(
                    "val_charts/q_error", np.mean(val_errors), global_step
                )
                writer.add_scalar(
                    "val_charts/q_error_norm", np.mean(val_errors_norm), global_step
                )
                writer.add_scalar(
                    "val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step
                )
                writer.add_scalar(
                    "val_charts/boredom", np.mean(val_boredom), global_step
                )
            with open(csv_path, "a") as csv_file:
                csv_file.write(f"val_charts/episodic_return,{np.mean(val_returns)},{global_step}\n")
            
            # csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns), global_step])
            # csv_writer.writerow(["val_charts/episodic_length", np.mean(val_lengths), global_step])
            # csv_writer.writerow(["val_charts/q_error", np.mean(val_errors), global_step])
            # csv_writer.writerow(["val_charts/q_error_norm", np.mean(val_errors_norm), global_step])
            # csv_writer.writerow(["val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step])
            # csv_writer.writerow(["val_charts/boredom", np.mean(val_boredom), global_step])
            # csv_file.flush()

        done = np.logical_or(terminated, truncated)
        if "final_info" in infos:
            if not args.observable:
                actor_state_encoder.reset()
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                if args.track == "wandb":
                    wandb.log(
                        {
                            "train_charts/episodic_return": info["episode"]["r"],
                            "train_charts/episodic_length": info["episode"]["l"],
                        },
                        global_step,
                    )
                elif args.track == "tensorboard":
                    writer.add_scalar(
                        "train_charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "train_charts/episodic_length", info["episode"]["l"], global_step
                    )
                
                with open(csv_path, "a") as csv_file:
                    csv_file.write(f"train_charts/episodic_return,{info['episode']['r']},{global_step}\n")
                    csv_file.flush()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, d in enumerate(infos["_final_observation"]):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
        if args.observable:
            obs_dict = {
                "obs": obs,
                "clicks": np.zeros((envs.num_envs, slate_size), dtype=np.int64), # Previous clicks are irrelevant
                "slate": -1 * np.ones((envs.num_envs, slate_size), dtype=np.int64), # Previous slate is irrelevant
            }
            next_obs_dict = {
                "obs": real_next_obs,
                "clicks": np.array(list(infos["clicks"])),
                "slate": effect_actions
            }
        else:
            obs_dict = obs # In the POMDP case, observations are already dictionaries
            next_obs_dict = real_next_obs
        hyper_actions = hyper_actions.cpu()
        rb.add(obs_dict, next_obs_dict, hyper_actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step == args.learning_starts:
            start_time = time.time()
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            if args.q_lr > 0:  # TD loss
                for _ in range(args.n_updates):
                    with torch.no_grad():
                        if args.observable:
                            actor_next_observations = qf_next_observations = data.next_observations["obs"].to(args.device)
                        else:
                            actor_next_observations = actor_state_encoder_target(data.next_observations).to(args.device)
                            qf_next_observations = qf_state_encoder_target(data.next_observations).to(args.device)

                        next_state_hyper_actions = actor_target.get_action(actor_next_observations)

                        qf_next_target = qf_target(
                            qf_next_observations,
                            next_state_hyper_actions,
                        )
                        next_q_value = data.rewards.flatten().to(args.device) + (
                            1 - data.dones.flatten().to(args.device)
                        ) * args.gamma * qf_next_target.view(-1)
                    if args.observable:
                        observations_qf = data.observations["obs"].to(args.device)
                    else:
                        observations_qf = qf_state_encoder(data.observations).to(args.device)
                    hyper_actions = actor.effect2hyper_map(data.next_observations["slate"].to(args.device)) # Estimated hyper-actions inferred from logged effect-actions
                    #qf_a_values = qf(data.observations["obs"], data.actions).view(-1)
                    qf_a_values = qf(observations_qf, hyper_actions).view(-1)
                    qf_loss = F.mse_loss(qf_a_values, next_q_value)

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()
            else:
                qf_a_values = torch.zeros((1,))
                qf_loss = torch.zeros((1,))

            if global_step % args.policy_frequency == 0 and args.policy_lr > 0:  # Actor loss
                for _ in range(
                    args.policy_frequency
                ):  # compensate  delay by doing 'policy_frequency' instead of 1
                    if args.observable:
                        observations_actor = observations_qf = data.observations["obs"].to(args.device)
                    else:
                        observations_actor = actor_state_encoder(data.observations).to(args.device)
                        observations_qf = qf_state_encoder(data.observations).to(args.device)
                    pi_hyper_actions = actor.get_action(observations_actor)
                    qf_pi = qf(observations_qf, pi_hyper_actions)
                    actor_loss = -qf_pi.mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
            else:
                actor_loss = torch.zeros((1,))

            if global_step % args.hyper_frequency == 0 and args.hyper_weight > 0 and args.policy_lr > 0:  # Hyper-actor loss
                for _ in range(
                    args.hyper_frequency
                ):  # compensate  delay by doing 'hyper_frequency' instead of 1
                    if args.observable:
                        observations_actor = data.observations["obs"].to(args.device)
                    else:
                        observations_actor = actor_state_encoder(data.observations).to(args.device)
                    hyper_actions = actor.get_action(observations_actor)
                    #hyper_actions = data.actions
                    est_hyper_actions = actor.effect2hyper_map(data.next_observations["slate"].to(args.device)) # Estimated hyper-actions inferred from policy effect-actions
                    hyper_loss = args.hyper_weight * F.mse_loss(est_hyper_actions, hyper_actions)

                    actor_optimizer.zero_grad()
                    hyper_loss.backward()
                    actor_optimizer.step()
            else:
                hyper_loss = torch.zeros((1,))

            if global_step % args.behavior_frequency == 0 and args.behavior_lr > 0:  # Behavior loss
                for _ in range(
                    args.behavior_frequency
                ):  # compensate  delay by doing 'behavior_frequency' instead of 1
                    if args.observable:
                        observations_actor = data.observations["obs"].to(args.device)
                    else:
                        observations_actor = actor_state_encoder(data.observations).to(args.device)
                    hyper_actions = actor.get_action(observations_actor)
                    item_scores = torch.einsum(
                        "i...j,i...kj->i...k",
                        hyper_actions,
                        actor.item_map(actor.item_features(data.next_observations["slate"].to(args.device)))
                    )
                    item_probs = torch.sigmoid(item_scores).to(args.device) # (num_envs, slate_size)
                    clicks = data.next_observations["clicks"].float().to(args.device) # (num_envs, slate_size)
                    clicks = torch.clamp(data.next_observations["clicks"].float().to(args.device), 0, 1)
                    behavior_loss = F.binary_cross_entropy(item_probs, clicks)

                    behavior_optimizer.zero_grad()
                    behavior_loss.backward()
                    behavior_optimizer.step()
            else:
                behavior_loss = torch.zeros((1,))

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf.parameters(), qf_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                if not args.observable:
                    for param, target_param in zip(
                        qf_state_encoder.parameters(), qf_state_encoder_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            # update the target actor
            if global_step % args.target_actor_frequency == 0:
                for param, target_param in zip(
                    actor.parameters(), actor_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                if not args.observable:
                    for param, target_param in zip(
                        actor_state_encoder.parameters(), actor_state_encoder_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            if global_step % 1000 == 0:
                if args.track == "wandb":
                    metric_dict = {
                        "train_charts/qf_values": qf_a_values.mean().item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                        "train_charts/SPS": int((global_step - args.learning_starts) / (time.time() - start_time)),
                    }
                    wandb.log(metric_dict, global_step)
                elif args.track == "tensorboard":
                    writer.add_scalar(
                        "train_charts/qf_values", qf_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("train_charts/SPS",
                        int((global_step - args.learning_starts) / (time.time() - start_time)),
                    )
                
                with open(csv_path, mode="a") as csv_file:
                    csv_file.write(f"train_charts/qf_values,{qf_a_values.mean().item()},{global_step}\n")
                    csv_file.write(f"losses/qf_loss,{qf_loss.item() / 2.0},{global_step}\n")
                    csv_file.write(f"losses/actor_loss,{actor_loss.item()},{global_step}\n")
                    csv_file.write(f"train_charts/SPS,{int((global_step - args.learning_starts) / (time.time() - start_time))},{global_step}\n")


                # csv_writer.writerow(["train_charts/qf_values", qf_a_values.mean().item(), global_step])
                # csv_writer.writerow(["losses/qf_loss", qf_loss.item() / 2.0, global_step])
                # csv_writer.writerow(["losses/actor_loss", actor_loss.item(), global_step])
                # csv_writer.writerow(["train_charts/SPS", int((global_step - args.learning_starts) / (time.time() - start_time)), global_step])

    envs.close()
    print(f"Elapsed time: {datetime.datetime.now() - start}")
    with open(csv_path2, "a") as csv_file:
        csv_file.write(f"\nElapsed time: {datetime.datetime.now() - start}\n")
    if args.track == "tensorboard":
        writer.close()
    csv_file.close()



def test(args):
    test_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, idx=0, observable=args.observable, args=args)])
    
    model_path = os.path.join("data", "topk_hac", f"actor_best_slatesize{args.slate_size}_numitem{args.num_items}_{args.seed}.pt")

    item_features = test_envs.envs[0].unwrapped.item_embedd
    slate_size = test_envs.envs[0].unwrapped.slate_size

    if args.raw_features:  # The latent space is the item raw feature space
        args.latent_dim = item_features.shape[1]
    
    actor = Actor(
        test_envs, args.hidden_size, args.state_dim, slate_size, args.latent_dim,
        args.latent_high, args.latent_low, args.reparam_std, item_features,
        args.raw_features
    ).to(args.device)

    # Load the saved model state
    actor.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    # Initialize the test environment
    test_obs, _ = test_envs.reset(seed=args.seed + 2)

    if not args.observable:
        if args.state_encoder == "gru":
            StateEncoder = GRUStateEncoder
        elif args.state_encoder == "transformer":
            StateEncoder = TransformerStateEncoder
        else:
            StateEncoder = None
        test_state_encoder = StateEncoder(test_envs, args).to(args.device)
        test_state_encoder.reset()

    # Run test episodes
    ep = 0
    cum_boredom = 0
    test_returns, test_lengths, test_diversity, test_catalog_coverage = [], [], [], []
    max_episodes = 500  # Number of test episodes to run
    while ep < max_episodes:
        with torch.no_grad():
            if args.observable:
                obs_tensor = torch.Tensor(test_obs).to(args.device)
            else:
                obs_tensor = test_state_encoder.step(test_obs).to(args.device)
            actions = actor.get_action(obs_tensor).cpu().numpy()

        # Take actions and get results
        next_obs, rewards, terminated, truncated, infos = test_envs.step(actions)
        test_obs = next_obs

        if "final_info" in infos:
            if not args.observable:
                test_state_encoder.reset()
            for info in infos["final_info"]:
                if info is None:
                    continue
                test_returns.append(info["episode"]["r"])
                test_diversity.append(info["diversity"])
                test_lengths.append(info["episode"]["l"])
                test_catalog_coverage.append(info["catalog_coverage"])
                ep += 1

    # Close the environment
    test_envs.close()

    # Print out test metrics
    print(f"Test Results over {max_episodes} Episodes:")
    print(f"Average Return: {np.mean(test_returns):.2f} ± {np.std(test_returns):.2f}")
    print(f"Average Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"Average Diversity: {np.mean(test_diversity):.2f} ± {np.std(test_diversity):.2f}")
    print(f"Average Catalog Coverage: {np.mean(test_catalog_coverage):.2f} ± {np.std(test_catalog_coverage):.2f}")

    with open(os.path.join("logs", f"hac_test_slatesize{args.slate_size}_num_items{args.num_items}_seed{str(args.seed)}-{DATE}.txt"), "w") as f:
        f.write(f"Average Return: {np.mean(test_returns):.2f} ± {np.std(test_returns):.2f}\n")
        f.write(f"Average Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}\n")
        f.write(f"Average Diversity: {np.mean(test_diversity):.2f} ± {np.std(test_diversity):.2f}\n")
        f.write(f"Average Catalog Coverage: {np.mean(test_catalog_coverage):.2f} ± {np.std(test_catalog_coverage):.2f}\n")


if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.track == "wandb":
        import wandb
        run_name = f"{args.env_id}__{args.run_name}__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
            monitor_gym=False,
            save_code=True,
        )

    train(args)