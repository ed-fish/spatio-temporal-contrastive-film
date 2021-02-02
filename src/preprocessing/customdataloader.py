from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

class TransVidDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame

    def __len__(self):
        return(len(self.data_frame))

    def __getitem__(self, idx):
        fp = self.data_frame.at[idx, "filepath"]
        name = self.data_frame.at[idx, "name"]
        scene = self.data_frame.at[idx, "scene"]
        genre = self.data_frame.at[idx, "genre"]
        data = self.data_frame.at[idx, "data"]
        sample = {"fp":fp, "name":name, "scene":scene, "genre":genre, "data":data}

        return sample
