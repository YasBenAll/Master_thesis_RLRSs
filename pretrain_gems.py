import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import time
import torch
import random
import numpy as np
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

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
        "--dataset-multi",
        type=str,
        default="SlateTopK-BoredInf-v0",
        help="Path to dataset for multi training",
    )

    return parser

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
    if args.multi:
        num_items = [100, 500, 1000]
        slate_sizes = [3, 5, 10, 20]
        kl_divergences = [0.1, 0.2, 0.5, 1.0, 2.0]
        lambda_clicks = [0.0, 0.3,0.5, 1.0]
        latent_dims = [16, 32]
        for latent_dim in latent_dims:
            for lambd in lambda_clicks:
                for beta in kl_divergences:
                    for num_item in num_items:
                        for slate_size in slate_sizes:
                            args.dataset=f"{args.dataset_multi}-num_item{num_item}-slate_size{slate_size}_oracle_epsilon0.5_seed2023_n_users100000seed2023.pt"
                            args.slate_size = slate_size
                            args.lambda_KL = beta
                            args.lambda_click = lambd
                            args.laten_dim = latent_dim
                            config_hash = hash_config(args, index=True)
                            print(f"Training GeMS with config: {config_hash}")
                            decoder = gems.train(args, config_hash)
    else:
        decoder = gems.train(args, config_hash)
    
    # set precision to 2 decimal places
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    # minutes 
    print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
    # hours
    print("--- %s hours ---" % round((time.time() - start_time)/3600, 2))
    print("### Finished pretraining GeMS ###")