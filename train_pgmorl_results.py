import argparse
import csv
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import sardine
import torch

from agents.wrappers import IdealState
from pymoo.indicators.hv import HV
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--slate-size", type=int, required=True, help="Size of the slate")
parser.add_argument("--num-items", type=int, required=True, help="Number of items")

args = parser.parse_args()

file_eval = os.path.join("data", "morl",f"pareto-archive-mosac-slatesize-{args.slate_size}-numitems{args.num_items}-timesteps500000-rankertopk-envsardine-SlateTopK-Bored-v0-evaluations.pkl")
file_catalog = os.path.join("data", "morl",f"pareto-archive-mosac-slatesize-{args.slate_size}-numitems{args.num_items}-timesteps500000-rankertopk-envsardine-SlateTopK-Bored-v0_catalog_coverage.pkl")


with open(file_eval, "rb") as f:
    evaluations = pickle.load(f)
    
with open(file_catalog, "rb") as f:
    catalog = pickle.load(f)

# print(evaluations)
# print(catalog)

import re 
for i, j in enumerate(zip(evaluations, catalog)):
    clicks = j[0][0]
    diversity = j[0][1]
    coverage = j[1]
    print(f"mosac_agent-{i},,sardine/SlateTopK-Bored-v0,{args.slate_size},{args.num_items},clicks,{clicks},,{diversity},,{coverage},")