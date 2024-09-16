import torch

with open('/home/yal700/test.txt', 'w') as f:
    f.write('Hello, World!')
    f.write(torch.cuda.is_available())
