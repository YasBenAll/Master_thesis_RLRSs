import torch

# wait 1 minute
import time
time.sleep(60)

with open('/home/yal700/test.txt', 'w') as f:
    f.write('Hello, World!')
    f.write(torch.cuda.is_available())
