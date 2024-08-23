import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import os
import pytorch_lightning as pl
import torch
import random
import numpy as np
from agents import sac
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--agent",
        type=str,
        required = True,
        choices=["sac", "ppo", "random"],
        help="Type of agent",
    )
    parser.add_argument(
        "--pretrain",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Whether to pretrain GeMS",
    )
    parser.add_argument(
        "--decoder-name",
        type=str,
        default="4910f9a5edb799495fcc6f154fe2ebf0cef4a44f6ceb59ce5e44a8d1ba093042",
        help="Name of the decoder",	
    )

    if args is not None:
        args, _ = parser.parse_known_args(args)
    else:
        args, _ = parser.parse_known_args()

    if args.pretrain:
        parser = gems.get_parser(parents = [parser])
    if args.agent == "sac":
        parser = sac.get_parser(parents = [parser])
    return parser

def main(parents = []):
    parser = get_parser(parents = parents)
    args = parser.parse_args()
    decoder = torch.load(os.path.join(args.data_dir, "GeMS", "decoder", args.exp_name,args.decoder_name+".pt"), map_location=torch.device('cpu')).to(args.device)

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

    if args.track == "wandb":
        import wandb
        run_name = f"{args.exp_name}_{args.run_name}_seed{args.seed}_{int(time.time())}"
        print("init wandb")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    print("### Training agent ###")

    if args.agent == "sac":
        sac.train(args, decoder = decoder)

if __name__ == "__main__":
    parser = get_generic_parser()
    main([parser])