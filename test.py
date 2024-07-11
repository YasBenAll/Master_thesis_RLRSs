import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())

array = np.zeros((10,3,3))
print(array)