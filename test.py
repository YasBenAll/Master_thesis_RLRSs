import torch

with open('/home/yal700/git/Master_thesis_RLRSs/test.txt', 'w') as f:
    f.write('Hello, World!\n')
    f.write(str(torch.cuda.is_available()))
