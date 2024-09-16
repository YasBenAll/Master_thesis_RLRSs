import torch

with open('test.txt', 'w') as f:
    f.write('Hello, World!')
    f.write(torch.cuda.is_available())
