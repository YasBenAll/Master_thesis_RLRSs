import argparse
from distutils.util import strtobool
from pathlib import Path
import os
import time
from typing import Dict, List, Union, NamedTuple

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from gems.models import GeMS
from utils.parser import get_generic_parser
from utils.file import hash_config

print(torch.__version__)
print(torch.cuda.is_available())