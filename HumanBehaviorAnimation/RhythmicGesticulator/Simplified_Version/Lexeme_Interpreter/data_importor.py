# region Import.

import numpy as np

from torch.utils.data import Dataset

# endregion


__all__ = ["TrainingDataset"]


class TrainingDataset(Dataset):
    def __init__(self, path_data: str) -> None:
        super().__init__()

        data = dict(np.load(path_data))

        self.audio_feat = data["audio"].astype(np.float32)  # num_clips X time X dim_feat.
        self.motion_feat = data["motion"].astype(np.float32)  # num_clips X time X dim_feat.
        self.index = data["index"].astype(int)  # num_clips X num_blocks.
        self.lexeme_index = data["lexeme_index"].astype(int)  # num_clips X num_blocks.

        self.max_index = int(np.max(self.index))

    def __len__(self):
        return self.audio_feat.shape[0]

    def __getitem__(self, idx):
        return {
            "audio": self.audio_feat[idx, :, :],
            "motion": self.motion_feat[idx, :, :],
            "lexeme_index": self.lexeme_index[idx, :],
            "index": self.index[idx, :]
        }