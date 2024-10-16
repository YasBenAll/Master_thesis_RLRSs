import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import os
import pytorch_lightning as pl
import torch
import random
import numpy as np
from agents import sac, ddpg, topk_reinforce, hac
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--agent",
        type=str,
        required = True,
        choices=["sac", "ppo", "random", "ddpg", "reinforce", "hac"],
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
        "--train",
        type=lambda x: bool(strtobool(x)),
        default=True,
    )
    parser.add_argument(
        "--test",
        type=lambda x: bool(strtobool(x)),
        default=False,
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="click",
        choices=["click", "diversity"],
    )

    if args is not None:
        args, _ = parser.parse_known_args(args)
    else:
        args, _ = parser.parse_known_args()

    if args.pretrain:
        parser = gems.get_parser(parents = [parser])
    if args.agent == "sac":
        parser = sac.get_parser(parents = [parser])
    if args.agent == "reinforce":
        parser = topk_reinforce.get_parser(parents = [parser])
    if args.agent == "hac":
        parser = hac.get_parser(parents = [parser])
    return parser

def main(parents = []):
    parser = get_parser(parents = parents)
    args = parser.parse_args()
    if args.agent == "sac":
        if args.ranker == "gems":
            decoder = torch.load(os.path.join(args.data_dir, "GeMS", "decoder", args.exp_name,args.decoder_name+".pt"), map_location=torch.device('cpu')).to(args.device)
        else: 
            decoder = None
    if args.ml100k: 
        args.num_items = 1682 # for logging purposes only as the number of items get set to 1682 in the simulator but not in the logging module
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
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

    if args.train:
        print("### Training agent ###")
        if args.agent == "sac":
            sac.train(args, decoder = decoder)
        if args.agent == "ddpg":
            ddpg.train(args)
        if args.agent == "reinforce":
            topk_reinforce.train(args)
        if args.agent == "hac":
            hac.train(args)
    if args.test:
        print("### Testing agent ###")
        if args.agent == "sac":
            sac.test(args, decoder = decoder)
        if args.agent == "ddpg":
            ddpg.test(args)
        if args.agent == "reinforce":
            topk_reinforce.test(args)
        if args.agent == "hac":
            hac.test(args)

if __name__ == "__main__":
    parser = get_generic_parser()
    main([parser])