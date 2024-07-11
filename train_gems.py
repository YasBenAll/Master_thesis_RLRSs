import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import random
import numpy as np
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    args, _ = parser.parse_known_args(args)
    return parser

if __name__ == "__main__":
    parser = get_parser(parents = [get_generic_parser(), gems.get_parser()])
    print(parser)
    args = parser.parse_args()
    decoder = None

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")


    print("### Pretraining GeMS ###")
    decoder_dir = args.data_dir + "GeMS/decoder/" + args.exp_name + "/"
    config_hash = hash_config(args)

    decoder = gems.train(args, config_hash)