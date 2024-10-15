import argparse
import os
from distutils.util import strtobool
from dotenv import load_dotenv

load_dotenv()

def get_generic_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.getenv("ROOT_DIR"),"data"),
        help="Path to data directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed.",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="sac",
        help="info on the run",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="final",
        help="the name of this experiment",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="final",
        help="the name of this run",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices = ["cuda", "cpu"],
        help="Device for training.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        choices = [None ,"wandb", "tensorboard"],
        help="Type of experiment tracking",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="SlateRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--morl",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to provide multiple rewards",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="engagement",
        choices=["engagement", "diversity", "novelty"],
        help="Type of reward",
    )
    parser.add_argument(
        "--env-embedds",
        type=str,
        default="item_embeddings_num_items",
    ),
    parser.add_argument(
        "--item-embeddings",
        type=str,
        default="ideal",
        choices=["ideal", "mf"],
    )
    parser.add_argument(
        "--decoder-name",
        type=str,
        default="SlateTopKBoredv0numitem100slatesize3_oracle_epsilon0.5_seed2023_n_users100000.ptkl_divergence1.0_lambda_click0.5_latentdim16.pt",
        help="Name of the decoder",	
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=100,
        help="Number of items in the environment.",
    )
    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Size of the slate.",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics in the dataset.",
    )
    parser.add_argument(
        "--ml100k",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use the ml100k dataset",
    )
    return parser