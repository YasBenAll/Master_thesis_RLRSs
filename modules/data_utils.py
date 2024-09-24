import torch
from tqdm import tqdm
from typing import Dict

class MFDataset(torch.utils.data.Dataset):
    '''
        Dataset used for the pre-training of item embeddings using Matrix Factorization.
    '''
    def __init__(self, data: Dict):
        self.data = []
        for u_id, user_traj in tqdm(data.items(), desc="Converting data to MF dataset"):
            slate_flat = user_traj["slate"].flatten()
            clicks_flat = user_traj["clicks"].flatten()
            for k, i_id in enumerate(slate_flat):
                if clicks_flat[k] == 1:
                    self.data.append((u_id, i_id.item()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
