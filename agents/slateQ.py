import argparse
import os
import csv
import random
import time
from distutils.util import strtobool
import sardine
import gymnasium as gym
import numpy as np

# import scipy.optimize as sco

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .buffer import ReplayBuffer, POMDPDictReplayBuffer
from .wrappers import IdealState, TopK, GeMS
from .state_encoders import GRUStateEncoder, TransformerStateEncoder

from utils.parser import get_generic_parser
from utils.file import hash_config, args2str

torch.set_float32_matmul_precision('high')

def get_parser(parents=[]):
    parser = argparse.ArgumentParser(parents=parents, add_help=False)
    # Training arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="ml-100k-v0",
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
        default=10000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1000,
        help="Number of timesteps between validation episodes.",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=1e3, help="timestep to start learning"
    )
    parser.add_argument(
        "--n-val-episodes",
        type=int,
        default=20,
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
        default=10,
        help="Number of timesteps to be sampled from replay buffer for each trajectory (only for POMDP)",
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="gems",
        choices=["topk", "gems"],
        help="Type of ranker for slate generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="the batch size of sample from the replay memory",
    )
    # SlateQ arguments
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
        "--q-lr",
        type=float,
        default=1e-3,
        help="the learning rate of the Q network optimizer",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=100,
        help="the frequency of updates for the target networks",
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
        "--state-encoder",
        type=str,
        default="gru",
        choices=["gru", "transformer"],
        help="Type of state encoder (only for POMDP)",
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
    # SlateQ-specific arguments
    parser.add_argument(
        "--opt-method",
        type=str,
        choices=["topk", "greedy", "lp"],
        default="topk",
        help="Optimization method for action selection in SlateQ",
    )
    parser.add_argument(
        "--rec-size",
        type=int,
        default=10,
        help="Recommendation slate size",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting value for epsilon in epsilon-greedy policy",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final value for epsilon in epsilon-greedy policy",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=10000,
        help="Number of steps over which epsilon decays",
    )
    return parser

def make_env(
    env_id,
    idx,
    observable,
    ranker,
    args,
    decoder,
):
    def thunk():
        env = gym.make(env_id, morl = args.morl)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if ranker == "topk":
            env = TopK(env, "ideal", min_action = 0, max_action = 1)
        elif ranker == "gems":
            env = GeMS(env,
                       path = args.data_dir + "GeMS/decoder/" + args.exp_name + "/" + '003753dba396f1ffac9969f66cd2f57e407dc14ba3729b2a1921fcbd8be577a4' + ".pt",
                       device = args.device,
                       decoder = decoder,
                    )
        if observable:
            env = IdealState(env)
        return env

    return thunk

class QNetwork(nn.Module):
    def __init__(self, env, hidden_size, state_dim):
        super().__init__()

        if state_dim is None:
            if isinstance(env.single_observation_space, gym.spaces.Dict):
                state_dim = sum(
                    np.prod(space.shape) for space in env.single_observation_space.spaces.values()
                )
            else:
                state_dim = np.prod(env.single_observation_space.shape)

        num_items = env.unwrapped.num_items  # Assuming your environment provides the number of items
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_items),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)  # Returns Q-values for all items

def get_action(q_network, state, epsilon, env, args):
    if np.random.rand() < epsilon:
        # Random action: select a random slate
        action = np.random.choice(
            env.unwrapped.num_items, size=args.rec_size, replace=False
        )
    else:
        with torch.no_grad():
            q_values = q_network(state)  # Shape: (num_items,)
            # Get item relevances from the environment
            relevances = torch.tensor(env.unwrapped.get_relevances(), device=args.device)
            # Adjust Q-values by relevances
            adjusted_q_values = q_values * relevances
            # Select top-K items
            action = torch.topk(adjusted_q_values, args.rec_size).indices.cpu().numpy()
    return action

def train(args):
    run_name = f"{args.env_id}__{args.run_name}__{args.seed}__{int(time.time())}"
    print("test")
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

    # CSV logger
    import datetime
    start = datetime.datetime.now()
    csv_filename = "slateq" + "-" +str(args.run_name) + "-" + str(datetime.datetime.now()) + "-seed" + str(args.seed) + ".log"
    # remove special characters
    import re
    csv_filename = re.sub(r"[^a-zA-Z0-9]+", '-', csv_filename)

    csv_path = "logs/" + csv_filename
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open (csv_path, "w") as f:
        f.write(f"Start: {start}\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Config: {args}\n")
        # write row
        f.write("field,value,step\n")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args.ranker,
                args,
                decoder=None,  # Assuming decoder is not needed
            )
        ]
    )
    val_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args.ranker,
                args,
                decoder=None,
            )
        ]
    )
    input(envs.single_action_space)
    assert isinstance(
        envs.single_action_space, gym.spaces.MultiDiscrete
    ), "action space must be MultiDiscrete"

    # Initialize Q-network and target network
    q_network = QNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
    target_q_network = QNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
    target_q_network.load_state_dict(q_network.state_dict())
    q_network_params = list(q_network.parameters())

    # Initialize state encoder if not observable
    if not args.observable:
        if args.state_encoder == "gru":
            StateEncoder = GRUStateEncoder
        elif args.state_encoder == "transformer":
            StateEncoder = TransformerStateEncoder
        else:
            StateEncoder = None
        state_encoder = StateEncoder(envs, args).to(args.device)
        q_network_params += list(state_encoder.parameters())

    optimizer = optim.Adam(q_network_params, lr=args.q_lr)

    # Set up the replay buffer
    action_space = gym.spaces.MultiDiscrete([envs.single_action_space.nvec[0]] * args.rec_size)
    if args.observable:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            action_space,
            args.device,
            handle_timeout_termination=True,
        )
    else:
        rb = POMDPDictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            action_space,
            args.sampled_seq_len,
            args.device,
            handle_timeout_termination=True,
        )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if not args.observable:
        state_encoder.reset()
    envs.single_action_space.seed(args.seed)

    for global_step in range(args.total_timesteps + 1):
        # Compute epsilon for epsilon-greedy policy
        epsilon = max(
            args.epsilon_end,
            args.epsilon_start - (global_step / args.epsilon_decay) * (args.epsilon_start - args.epsilon_end),
        )

        # Select action using epsilon-greedy policy
        if args.observable:
            state = torch.tensor(obs, dtype=torch.float32, device=args.device).squeeze(0)
        else:
            state = state_encoder.step(obs)

        action = get_action(q_network, state, epsilon, envs.envs[0], args)

        # Execute the action in the environment
        next_obs, rewards, terminated, truncated, infos = envs.step([action])

        # Store the transition in the replay buffer
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, d in enumerate(infos["_final_observation"]):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, action, rewards, terminated, infos)

        obs = next_obs

        # ALGO LOGIC: training
        if global_step == args.learning_starts:
            start_time = time.time()
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            if args.observable:
                state = torch.tensor(data.observations, dtype=torch.float32, device=args.device)
                next_state = torch.tensor(data.next_observations, dtype=torch.float32, device=args.device)
            else:
                state = state_encoder(data.observations)
                next_state = state_encoder(data.next_observations)

            # Q-values for current state
            q_values = q_network(state)  # Shape: (batch_size, num_items)

            # Extract the actions taken (slates)
            actions = torch.tensor(data.actions, dtype=torch.long, device=args.device)  # Shape: (batch_size, rec_size)

            # The environment should provide 'clicked_item' in infos
            clicked_items = torch.tensor([info.get('clicked_item', -1) for info in data.infos], dtype=torch.long, device=args.device)
            # Create a mask for samples where there was a click
            click_mask = (clicked_items != -1)

            # Gather Q-values for clicked items
            q_values_clicked = q_values[torch.arange(q_values.size(0)), clicked_items.clip(min=0)]

            with torch.no_grad():
                next_q_values = target_q_network(next_state)  # Shape: (batch_size, num_items)
                # Get item relevances from the environment
                relevances = torch.tensor(envs.envs[0].get_relevances(), device=args.device)
                adjusted_next_q_values = next_q_values * relevances

                # For each sample in the batch, compute the next action (slate)
                next_actions = torch.topk(adjusted_next_q_values, args.rec_size, dim=1).indices  # Shape: (batch_size, rec_size)

                # Compute attractiveness and click probabilities
                attractiveness = relevances[next_actions]  # Shape: (batch_size, rec_size)
                click_probs = envs.envs[0].compute_click_probs(attractiveness)  # Shape: (batch_size, rec_size)

                # Compute expected Q-values
                expected_next_q = (next_q_values.gather(1, next_actions) * click_probs).sum(dim=1)

                target_q_values = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * expected_next_q

            # Only compute loss for samples where there was a click
            loss = F.mse_loss(q_values_clicked[click_mask], target_q_values[click_mask])

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging
            if global_step % args.val_interval == 0:
                print(f"Global Step: {global_step}, Loss: {loss.item():.4f}")
                if args.track == "wandb":
                    wandb.log({"loss": loss.item()}, step=global_step)
                elif args.track == "tensorboard":
                    writer.add_scalar("loss", loss.item(), global_step)
                with open(csv_path, "a") as csv_file:
                    csv_file.write(f"loss,{loss.item()},{global_step}\n")

    envs.close()
    if args.track == "tensorboard":
        writer.close()
    with open(csv_path, "a") as csv_file:
        csv_file.write(f"Elapsed time: {datetime.datetime.now() - start}\n")
    if args.track == "wandb":
        wandb.finish()

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
    args.device = device

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

    train_slateq(args)
