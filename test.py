import torch

print("Output python file:")
print(f"Torch version: {torch.__version__}")
print(f"Is Cuda available: {str(torch.cuda.is_available())}")

with open('/home/yal700/git/Master_thesis_RLRSs/test.txt', 'w') as f:
    f.write(f"Torch version: {torch.__version__}")
    f.write(f"Is Cuda available: {str(torch.cuda.is_available())}")

