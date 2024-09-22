import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import concurrent.futures
import pytorch_lightning as pl
import time
import torch
import random
import re
import numpy as np
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config
import cProfile
import pstats
import io

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    args, _ = parser.parse_known_args(args)
    parser.add_argument(
        "--multi",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to train multiple configurations",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="SlateTopK-BoredInf-v0",
        help="environment id",
    )
    parser.add_argument(
        "--concurrent",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Number of users to generate data for",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=100,
        help="Number of items in the dataset",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=100000,
        help="Number of users in the dataset",
    )

    return parser

def train_gems(args, num_item, slate_size, lambd, latent_dim, env_id):
    """Function to train GeMS with a specific configuration."""
    args.dataset = f"{env_id}numitem{num_item}slatesize{slate_size}_oracle_epsilon0.5_seed2023_n_users100.pt"
    args.slate_size = slate_size
    args.lambda_click = lambd
    args.latent_dim = latent_dim
    config_hash = hash_config(args, index=True)
    print(f"Training GeMS with configurations: args.dataset={args.dataset}, args.slate_size={args.slate_size}, args.lambda_KL={args.lambda_KL}, args.lambda_click={args.lambda_click}, args.latent_dim={args.latent_dim}")

    decoder = gems.train(args, config_hash)
    return config_hash

if __name__ == "__main__":
    parser = get_parser(parents = [get_generic_parser(), gems.get_parser()])
    args = parser.parse_args()
    start_time = time.time()

    config_hash = hash_config(args, index=True)
    decoder_dir = args.data_dir + "GeMS/decoder/" + args.exp_name + "/"
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    
    print("### Pretraining GeMS ###")
    dataset_name = args.dataset
    # num_items = [100, 500, 1000]
    num_items = [100,500, 1000]
    # slate_sizes = [5]
    # slate_sizes = [3, 5, 10, 20]
    # kl_divergences = [0.1, 0.2, 0.5, 1.0, 2.0]
    kl_divergences = [0.5]
    # lambda_clicks = [0.0, 0.3,0.5, 1.0]
    lambda_clicks = [0.0, 0.3, 0.5, 1.0]
    # latent_dims = [16, 32]
    latent_dims = [16]

    # number of combinations of hyperparameters
    print(f"Number of combinations: {len(num_items) * len(slate_sizes) * len(lambda_clicks) * len(latent_dims)}")
    total = len(num_items) * len(slate_sizes) * len(lambda_clicks) * len(latent_dims)
    env_id = re.sub(r'[\W_]+', '', args.env_id)
    if args.multi:
        # Use ThreadPoolExecutor or ProcessPoolExecutor for concurrent executionp
        if args.concurrent: 
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                futures = []
                for latent_dim in latent_dims:
                    for lambd in lambda_clicks:
                        for num_item in num_items:
                            # for slate_size in slate_sizes:
                            # Submit each configuration to the executor for parallel processing
                            futures.append(executor.submit(train_gems, args, num_item, args.slate_size, lambd, latent_dim, env_id))

                # Collect results (if necessary)
                for future in concurrent.futures.as_completed(futures):
                    config_hash = future.result()
                    print(f"Completed training with config: {config_hash}")
        else:
            for latent_dim in latent_dims:
                for lambd in lambda_clicks:
                    for num_item in num_items:
                        for slate_size in slate_sizes:
                            args.dataset=f"{env_id}numitem{num_item}slatesize{slate_size}_oracle_epsilon0.5_seed2023_n_users{args.n_users}.pt"
                            args.slate_size = slate_size
                            args.lambda_KL = args.lambda_KL
                            args.lambda_click = lambd
                            args.latent_dim = latent_dim
                            config_hash = hash_config(args, index=True)
                            print(f"Training GeMS with configurations: args.dataset={args.dataset}, args.slate_size={args.slate_size}, args.lambda_KL={args.lambda_KL}, args.lambda_click={args.lambda_click}, args.latent_dim={args.latent_dim}")
                            decoder = gems.train(args, config_hash)
    else:
        args.dataset = f"{env_id}numitem{args.num_items}slatesize{args.slate_size}_oracle_epsilon0.5_seed2023_n_users{args.n_users}.pt"
        decoder = gems.train(args, config_hash)
        
    import os
    if args.multi:
        log_folder = os.path.join("pretrain_gems", f"{env_id}numitem{num_items}slatesize{args.slate_size}_oracle_epsilon0.5_seed2023_n_users{args.n_users}.pt"[:-3]+".log")
    else:
        log_folder = os.path.join("pretrain_gems", f"{env_id}numitem{args.num_items}slatesize{args.slate_size}_oracle_epsilon0.5_seed2023_n_users{args.n_users}.pt"[:-3]+".log")
    with open(log_folder, "w") as f:
        f.write(f"seed:{args.seed}\n")
        f.write(f"env_id:{args.env_id}\n")
        f.write(f"n_users:{args.n_users}\n")
        f.write("\nconfigurations trained: \n")
        if args.multi:
            for latent_dim in latent_dims:
                for lambd in lambda_clicks:
                    for num_item in num_items:
                        for slate_size in slate_sizes:
                            f.write(f"num_item: {num_item}, slate_size: {slate_size}, lambda_click: {lambd}, latent_dim: {latent_dim}\n")
            f.write(f"\nconcurrent: {args.concurrent}\n")
            f.write(f"total configurations: {total}\n")
            f.write(f"average training time per configuration: {round((time.time() - start_time)/total, 2)} seconds\n")
        else:
            f.write(f"num_item: {args.num_items}, slate_size: {args.slate_size}, lambda_click: {args.lambda_click}, latent_dim: {args.latent_dim}\n")
            f.write(f"total configurations: 1\n")
        
        f.write(f"total time: {round(time.time() - start_time, 2)} seconds\n")



    # set precision to 2 decimal places
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    # minutes 
    print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
    # hours
    print("--- %s hours ---" % round((time.time() - start_time)/3600, 2))

    print("Average time per configuration: ", round((time.time() - start_time)/total, 2), "seconds")
    print("### Finished pretraining GeMS ###")