"""Code for module used taken from https://github.com/naver/gems/tree/master/GeMS/modules/MatrixFactorization"""

import os
import torch
import pytorch_lightning as pl
import random

from modules.argument_parser import MainParser
from modules.item_embeddings import MFEmbeddings

argparser = MainParser() # Program-wide parameters
argparser = MFEmbeddings.add_model_specific_args(argparser)  # Agent-specific parameters
args = argparser.parse_args()
arg_dict = vars(args)

# Seeds for reproducibility
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

numitems = [100, 500, 1000]
slatesizes = [3, 5, 10, 20]
for numitem in numitems:
    for slatesize in slatesizes:
        print(f"Training for numitem: {numitem} and slatesize: {slatesize}")
        dataset = f"SlateTopKBoredv0numitem{numitem}slatesize{slatesize}_oracle_epsilon0.5_seed2023_n_users100000"
        arg_dict["dataset_name"] = dataset
        arg_dict["device"]="cuda"
        arg_dict["num_items"] = numitem
        dataset_path = os.path.join("data", "datasets", f"{dataset}.pt")

        item_embeddings = MFEmbeddings(**arg_dict)

        import time 

        start = time.time()
        item_embeddings.train(dataset_path, os.path.join("data", "mf_embeddings"))
        print(f"Training time: {round(time.time() - start,2 )} seconds")
        print(f"Finished training for {dataset}")