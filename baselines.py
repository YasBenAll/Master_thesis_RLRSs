import gymnasium as gym
import numpy as np
import os
import csv
import torch
import sardine
import argparse
import datetime
from agents.wrappers import IdealState
from pymoo.indicators.hv import HV

def parse_args():
    parser = argparse.ArgumentParser(description="Environment Evaluation")
    parser.add_argument("--morl", action="store_true", help="Specify if the algorithm is multi-objective.")
    parser.add_argument("--env-ids", type=str, nargs='+', default=["sardine/SlateTopK-Bored-v0"], help="List of environment IDs.")
    parser.add_argument("--methods", type=str, nargs='+', default=["random"], help="List of methods to evaluate.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[2705, 3751, 4685, 3688, 6383], help="List of seeds for reproducibility.")
    parser.add_argument("--n-val-episodes", type=int, default=200, help="Number of validation episodes.")
    parser.add_argument("--total-timesteps", type=int, default=10000, help="Total timesteps for the evaluation.")
    parser.add_argument("--val-interval", type=int, default=100, help="Interval between validation steps.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    start = datetime.datetime.now()
    click_list = []
    diversity_list = []
    hypervolume_list = []
    num_users = 0

    for env_id in args.env_ids:
        # Make the environment
        env = gym.make(env_id)
        env_name = "-".join(env_id.lower().split("-")[:-1])

        for method in args.methods:
            val_returns_all = []
            for seed in args.seeds:
                # CSV logger
                csv_filename = f"run_{env_name}-{method}-0-{seed}.log"
                csv_path = os.path.join(args.log_dir, csv_filename)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                csv_file = open(csv_path, "w+", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["field", "value", "step"])

                # Environment initialization
                observation, info = env.reset(seed=seed)
                env.action_space.seed(seed)

                # Run the agent on the environment
                val_returns, val_boredom = [], []
                seed_clicks = []
                seed_diversity = []
                
                for _ in range(args.n_val_episodes):
                    cum_reward = np.array([0., 0.]) if args.morl else np.array([0.])
                    cum_boredom = 0
                    terminated, truncated = False, False
                    num_users += 1
                    diversity = 0
                    
                    while not (terminated or truncated):
                        if method == "greedyoracle":  # Greedy oracle
                            action = -np.ones(env.unwrapped.slate_size, dtype=int)
                        elif method == "random":  # Random slate
                            action = torch.randint(low=0, high=env.unwrapped.num_items, size=(env.unwrapped.slate_size,)).numpy()

                        observation, reward, terminated, truncated, info = env.step(action)
                        cum_reward += reward
                        cum_boredom += (1.0 if np.sum(info.get("bored", False)) > 0 else 0.0)
                        diversity += info["diversity"]

                        if terminated or truncated:
                            observation, info = env.reset()

                    val_returns.append(cum_reward)
                    val_boredom.append(cum_boredom)
                    seed_clicks.append(cum_reward[0])
                    if args.morl:
                        seed_diversity.append(cum_reward[1])
                    else:
                        seed_diversity.append(diversity/env.unwrapped.H)
                        
                click_list.append(seed_clicks)
                if args.morl:
                    hypervolume_list.append(HV(ref_point=np.array([0.0, 0.0]) * -1)(val_returns * -1))

                diversity_list.append(seed_diversity)

                val_returns_all += val_returns

                # Log the results
                for i in range(args.total_timesteps // args.val_interval + 1):
                    global_step = i * args.val_interval  # Simulate a global step as in trained agents for plotting purposes
                    csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns, axis=0), global_step])
                    csv_writer.writerow(["val_charts/boredom", np.mean(val_boredom), global_step])

                print(f"{method} -- {seed} -- summary: cum_reward = {np.mean(val_returns, axis=0)}, diversity = {np.mean(seed_diversity)}")

        env.close()

    end = datetime.datetime.now()

    print("Time taken: ", end-start)
    print(f"Number of users: {num_users}")

    mean_click_list = np.mean(np.array(click_list), axis=1)
    mean_diversity_list = np.mean(np.array(diversity_list), axis=1)
    print(f"Mean click list: {np.mean(mean_click_list)}+-{np.std(mean_click_list)}")
    print(f"Mean diversity list: {np.mean(mean_diversity_list)}+-{np.std(mean_diversity_list)}")
    if args.morl:
        print(f"Mean hypervolume: {np.mean(hypervolume_list)}+-{np.std(hypervolume_list)}")

if __name__ == "__main__":
    main()
