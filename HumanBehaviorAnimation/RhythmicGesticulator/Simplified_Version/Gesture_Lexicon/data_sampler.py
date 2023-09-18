# region Import.

import numpy as np

from torch.utils.data import Dataset

# endregion


__all__ = ["TrainingDataset"]


class TrainingDataset(Dataset):
    def __init__(self, path_data : str) -> None:
        super().__init__()
        
        data = np.load(path_data)
        
        self.motion_feat = data["motion"].astype(np.float32)  # num_clips X time X dim_feat.
        self.index = data["index"].astype(int)  # num_clips X num_blocks.
        
        self.motion_block = np.concatenate(np.split(self.motion_feat, self.index.shape[1], axis=1), axis=0)
        self.motion_block = np.transpose(self.motion_block, (0, 2, 1))  # (num_clips*num_blocks) X dim_feat X time.
        self.index_new = np.concatenate(np.split(self.index, self.index.shape[1], axis=1), axis=0).reshape(-1)
        
        # print(self.motion_block.shape)
        # print(self.index_new.shape)
    
    def __len__(self):
        return self.index.shape[0] * self.index.shape[1]
    
    def __getitem__(self, idx):
        return {
            "motion": self.motion_block[idx, :, :],
            "index": self.index_new[idx]
        }