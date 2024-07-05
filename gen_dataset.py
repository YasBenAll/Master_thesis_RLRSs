import argparse
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import os
import pickle
import sardine
from momdp.momdp import MOMDP
from momdp.policy import create_policy
from momdp.evaluation import evaluate_momdp
from pathlib import Path
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle, EpsilonGreedyAntiOracle, RandomPolicy
from collections import OrderedDict
np.set_printoptions(precision=3)

def get_parser(parents  = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help=False)
    parser.add_argument('--env_id', type=str, default='ml-100k-v0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--lp', type=str, default='oracle')
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument("--n_users", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset_type", type=str, default="dict")
    return parser

def generate_dataset(env, seed, lp, eps, n_users, dataset_type):

    logging_policy = create_policy(lp=lp, epsilon=eps, env=env, seed=seed)
    dataset = env.unwrapped.generate_dataset(n_users=n_users, policy=logging_policy, seed=seed, dataset_type=dataset_type)
    return dataset

def create_policy(lp, epsilon, env, seed):
    if lp == "oracle":
        return EpsilonGreedyOracle(epsilon = epsilon, env = env, seed = seed)
    elif lp == "antioracle":
        return EpsilonGreedyAntiOracle(epsilon = epsilon, env = env, seed = seed)
    elif lp == "random":
        return RandomPolicy(env = env, seed = seed)
    else:
        raise ValueError("Unknown logging policy")

def evaluate(env, seed):
    observation, info = env.reset(seed = seed)
    env.action_space.seed(2024)
    cum_reward_list, cum_boredom_list = [], []

    cum_reward = np.zeros(3)  # For engagement, diversity, and novelty
    cum_boredom = 0
    ep = 0


    while ep < 100:
        # print(ep)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        clicks = info["clicks"]
        slate = info["slate"]
        rewards = np.array([reward["engagement"], reward["diversity"], reward["novelty"]])
        cum_reward += rewards
        cum_boredom += (1.0 if np.sum(info["bored"] == True) > 0 else 0.0)

        if terminated or truncated:
            cum_reward_list.append(cum_reward)
            cum_boredom_list.append(cum_boredom)
            cum_reward = 0
            cum_boredom = 0
            observation, info = env.reset()
            ep += 1
    # print(cum_reward_list)
    print("Average return: ", np.mean(cum_reward_list, axis=0))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir + "datasets/").mkdir(parents=True, exist_ok=True)

    env = mo_gym.make(f"sardine/{args.env_id}")
    env = IdealState(env)

    dataset = generate_dataset(env, args.seed, args.lp, args.eps, args.n_users, args.dataset_type)
    print(dataset)
    # save ordered dict
    path = os.path.join("data", "datasets", f"env_data_{args.env_id}_lp_{args.lp}_epsilon_{args.eps}_seed_{args.seed}_n_users_{args.n_users}")

    if type(dataset) == OrderedDict:
        with open(f'{path}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            print("Saved dataset to ", path)
    elif type(dataset) == "sb3_replay":
        dataset.save(path)
        print("Saved dataset to ", path)

    print("Done!")