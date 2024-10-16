import gymnasium as gym
import numpy as np
import os
import csv
import torch
import sardine
import re
import argparse
import datetime
from agents.wrappers import IdealState
from pymoo.indicators.hv import HV
from distutils.util import strtobool


from agents.plots import plot

def parse_args():
    parser = argparse.ArgumentParser(description="Environment Evaluation")
    parser.add_argument("--morl", type=lambda x: bool(strtobool(x)), default=False, help="Specify if the algorithm is multi-objective.")
    parser.add_argument("--env-ids", type=str, nargs='+', default=["sardine/SlateTopK-Bored-v0"], help="List of environment IDs.")
    parser.add_argument("--methods", type=str, nargs='+', default=["random"], help="List of methods to evaluate.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[2705, 3751, 4685, 3688, 6383], help="List of seeds for reproducibility.")
    parser.add_argument("--n-val-episodes", type=int, default=500, help="Number of validation episodes.")
    parser.add_argument("--total-timesteps", type=int, default=10000, help="Total timesteps for the evaluation.")
    parser.add_argument("--val-interval", type=int, default=100, help="Interval between validation steps.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--slate-size", type=int, default=10, help="Size of the slate.")
    parser.add_argument("--ml100k", type=lambda x: bool(strtobool(x)), default=False, help="Specify if the environment is MovieLens-100K.")
    
    return parser.parse_args()

def log_data(method, steps, returns, log_dir):
    """
    Logs the method returns data into a CSV file.
    """
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"{method}_returns.csv")
    
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing header
        csvwriter.writerow(["steps", "return"])
        # Writing data
        for step, ret in zip(steps, returns):
            csvwriter.writerow([step, ret])

def main():
    args = parse_args()

    slate_list = [1, 3, 10, 20]
    num_items = [1682]

    slate_dict = {}
    for slate_size in slate_list:
        for num_item in num_items:
            
            for env_id in args.env_ids:
                # Make the environment
                env = gym.make(env_id, slate_size=slate_size, morl=args.morl, num_items=num_item, ml100k = args.ml100k)
                env_name = "-".join(env_id.lower().split("-")[:-1])
                env_dict = {}
                scores_dict = {}
                for method in args.methods:
                    print("method: ", method)
                    print("slate size: ", slate_size)
                    print("num items: ", num_item)	
                    click_list = []
                    diversity_list = []
                    hypervolume_list = []
                    catalog_coverage_list = []  # New list to track catalog coverage
                    num_users = 0
                    val_returns_all = []
                    seed_dict = {}

                    methods_dict = {}
                    relevances = []
                    user_embedds = []

                    runtime_list = []
                    for seed in args.seeds:
                        start = datetime.datetime.now()
                        info_dict = {}
                        # CSV logger
                        csv_filename = f"run_{env_name}-{method}-0-{seed}"
                        csv_filename = re.sub(r"[^a-zA-Z0-9]+", '-', csv_filename)
                        csv_path = os.path.join(args.log_dir, "baselines", csv_filename + ".log")
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        csv_file = open(csv_path, "w+", newline="")
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["field", "value", "step"])

                        # Environment initialization
                        observation, info = env.reset(seed=seed)
                        env.action_space.seed(seed)

                        # Run the agent on the environment
                        val_returns, val_diversity, val_catalog_coverage = [], [], []
                        seed_clicks = []
                        seed_catalog_coverage = []
                        
                        val_metrics = []
                        for _ in range(args.n_val_episodes):
                            cum_reward = 0
                            cum_boredom = 0
                            terminated, truncated = False, False
                            num_users += 1
                            diversity = 0

                            val_reward_list = []
                            catalog_coverage = []
                            val_diversity_list = []
                            while not (terminated or truncated):
                                action = None

                                if method == "greedyoracle":  # Greedy oracle
                                    action = -np.ones(env.unwrapped.slate_size, dtype=int)
                                elif method == "random":  # Random slate
                                    action = torch.randint(low=0, high=env.unwrapped.num_items, size=(env.unwrapped.slate_size,)).numpy()
                                else: 
                                    raise ValueError(f"Unknown method: {method}")
                                observation, reward, terminated, truncated, info = env.step(action)
                                
                                cum_reward += reward
                                cum_boredom += (1.0 if np.sum(info.get("bored", False)) > 0 else 0.0)
                                diversity += info["diversity"]

                                val_reward_list.append(reward)
                                val_diversity_list.append(info["diversity"])
                                catalog_coverage.append(info["catalog_coverage"])  # Track catalog coverage
                                relevances.extend(list(info["scores"]))

                                if terminated or truncated:
                                    user_embedds.append(env.unwrapped.user_embedd)
                                    val_metrics.append({"clicks": val_reward_list, "diversity": val_diversity_list, "catalog_coverage": catalog_coverage})  # Store the episode metrics
                                    observation, info = env.reset()

                            val_returns.append(cum_reward)
                            seed_clicks.append(cum_reward)
                            seed_catalog_coverage.append(catalog_coverage[-1])  # Store the last catalog coverage value for this episode
                            if args.morl:
                                val_diversity.append(cum_reward[1])
                            else:
                                val_diversity.append(diversity / env.unwrapped.H)
                            val_catalog_coverage.append(np.mean(catalog_coverage))  # Store average catalog coverage per episode

                        seed_dict[seed] = val_metrics
                        click_list.append(seed_clicks)
                        catalog_coverage_list.append(seed_catalog_coverage)  # Add catalog coverage to the list
                        
                        if args.morl:
                            hypervolume_list.append(HV(ref_point=np.array([0.0, 0.0]) * -1)(val_returns * -1))

                        diversity_list.append(val_diversity)
                        val_returns_all += val_returns

                        # Log the results (including catalog coverage)

                        for i in range(args.total_timesteps // args.val_interval + 1):
                            global_step = i * args.val_interval  # Simulate a global step as in trained agents for plotting purposes
                            csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns, axis=0), global_step])
                            csv_writer.writerow(["val_charts/diversity", np.mean(val_diversity), global_step])
                            csv_writer.writerow(["val_charts/catalog_coverage", np.mean(val_catalog_coverage), global_step])  # New log for catalog coverage

                        print(f"{method} -- {seed} -- summary: cum_reward = {np.mean(val_returns, axis=0)}, diversity = {np.mean(val_diversity)}, catalog_coverage = {np.mean(val_catalog_coverage)}")
                        info_dict["clicks"]=val_returns
                        info_dict["diversity"]=val_diversity
                        info_dict["catalog_coverage"]=val_catalog_coverage
                        runtime_list.append((datetime.datetime.now() - start).total_seconds())
            
                    mean_click_list = np.mean(np.array(click_list), axis=1)
                    mean_diversity_list = np.mean(np.array(diversity_list), axis=1)
                    mean_catalog_coverage_list = np.mean(np.array(catalog_coverage_list), axis=1)  # Mean catalog coverage
                    print(f"Mean click list: {np.mean(mean_click_list):.2f}+-{np.std(mean_click_list):.2f}")
                    print(f"Mean diversity list: {np.mean(mean_diversity_list):.2f}+-{np.std(mean_diversity_list):.2f}")
                    print(f"Mean catalog coverage: {np.mean(mean_catalog_coverage_list):.2f}+-{np.std(mean_catalog_coverage_list):.2f}")  # New print for catalog coverage
                    print(f"Elapsed time: {np.mean(runtime_list):.2f} +- {np.std(runtime_list):.2f} seconds")
                    
                    with open(os.path.join("results", "results.csv"), "a") as f:
                        f.write(f"\n{method},{''},{env_id},{slate_size},{num_item},{''},{np.mean(mean_click_list):.2f},{np.std(mean_click_list):.2f},{np.mean(mean_diversity_list):.2f},{np.std(mean_diversity_list):.2f},{np.mean(mean_catalog_coverage_list):.2f},{np.std(mean_catalog_coverage_list):.2f},{np.mean(runtime_list):.2f},{np.std(runtime_list):.2f}")







                    # from sklearn.decomposition import PCA
                    # import matplotlib.pyplot as plt
                    # def plot_user_embedding_distribution(user_embeddings):
                    #     """
                    #     Plot the distribution of user embeddings.

                    #     Arguments:
                    #     - user_embeddings: A NumPy array of shape (num_users, num_topics) containing user embeddings.
                    #     """

                    #     num_topics = user_embeddings.shape[1]

                    #     # Plot histograms for each topic to show their distribution
                    #     fig, axs = plt.subplots(1, num_topics, figsize=(20, 5))
                    #     for i in range(num_topics):
                    #         axs[i].hist(user_embeddings[:, i], bins=20, alpha=0.7, color='blue')
                    #         axs[i].set_title(f'Topic {i+1}')
                    #         axs[i].set_xlabel('Embedding Value')
                    #         axs[i].set_ylabel('Frequency')

                    #     plt.tight_layout()
                    #     plt.suptitle('User Embedding Value Distributions by Topic', y=1.02, fontsize=16)
                    #     plt.show()

                    #     # Perform PCA to reduce the dimensionality of embeddings to 2D for visualization
                    #     pca = PCA(n_components=2)
                    #     reduced_embeddings = pca.fit_transform(user_embeddings)

                    #     # Scatter plot of reduced embeddings
                    #     plt.figure(figsize=(10, 7))
                    #     plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, color='red')
                    #     plt.xlabel('PCA Component 1')
                    #     plt.ylabel('PCA Component 2')
                    #     plt.title('2D Projection of User Embeddings (PCA)')
                    #     plt.grid(True)
                    #     plt.show()

                    # # Example usage
                    # # Replace this with actual user embeddings obtained from the simulator
                    # # For demonstration, we generate random embeddings for 100 users and 10 topics
                    # plot_user_embedding_distribution(np.array(user_embedds))


                    scores_dict[method] = relevances

                    relevances = []
                    methods_dict["clicks"] = {"value":np.mean(mean_click_list), "std":np.std(mean_click_list)}
                    methods_dict["diversity"] = {"value":np.mean(mean_diversity_list), "std":np.std(mean_diversity_list)}
                    methods_dict["catalog_coverage"] = {"value":np.mean(mean_catalog_coverage_list), "std":np.std(mean_catalog_coverage_list)}  # New log for catalog coverage
                


                    if args.morl:
                        print(f"Mean hypervolume: {np.mean(hypervolume_list)}+-{np.std(hypervolume_list)}")

                    env_dict[method] = methods_dict  
                # plot.plot_violin(scores_dict, env_id)
            slate_dict[slate_size] = env_dict

        env.close()


    print(slate_dict)  
    end = datetime.datetime.now()

    print("Time taken: ", end - start)
    print(f"Number of users: {num_users}")



if __name__ == "__main__":
    main()