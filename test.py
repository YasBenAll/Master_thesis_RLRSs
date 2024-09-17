import torch

print("testestest")
print(torch.__version__)
with open('/home/yal700/git/Master_thesis_RLRSs/test.txt', 'w') as f:
    f.write('Hello, World!\n')
    f.write(torch.__version__ + '\n')
    f.write(str(torch.cuda.is_available()))

