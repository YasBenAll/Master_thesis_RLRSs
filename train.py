import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import random
import numpy as np
from agents import sac, pgmorl
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--agent",
        type=str,
        required = True,
        choices=["sac", "ddpg", "hac", "reinforce", "topk_reinforce", "pgmorl"],
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

    if args is not None:
        args, _ = parser.parse_known_args(args)
    else:
        args, _ = parser.parse_known_args()

    if args.agent == "sac":
        parser = sac.get_parser(parents = [parser])
    elif args.agent == "pgmorl":
        parser = pgmorl.get_parser(parents = [parser])

    return parser

def main(parents = []):
    parser = get_parser(parents = parents)
    args = parser.parse_args()
    decoder = None

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print("### Training agent ###")
    if args.agent == "sac":
        sac.train(args, decoder = decoder)
    elif args.agent == "pgmorl":
        pgmorl.train(args, decoder=decoder)

if __name__ == "__main__":
    parser = get_generic_parser()
    main([parser])
