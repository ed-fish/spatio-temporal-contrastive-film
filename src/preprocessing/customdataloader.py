from __future__ import print_function, division
from torch.utils.data import Dataset
import warnings
import random
import pickle
import torch
import pandas as pd

# For supervised training
class TransVidDataset(Dataset):
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fp = self.data_frame.at[idx, "filepath"]
        name = self.data_frame.at[idx, "name"]
        scene = self.data_frame.at[idx, "scene"]
        genre = self.data_frame.at[idx, "genre"]
        data = self.data_frame.at[idx, "data"]
        sample = {"fp": fp, "name": name, "scene": scene, "genre": genre, "data": data}

        return sample


class DataLoader:
    def __init__(self, data_fp, sample_size, supervised=False):
        self.supervised = supervised
        self.__load_data__(data_fp, sample_size)

    def __load_data__(self, data_fp, sample_size):
        data_list = []
        pklfl = open(data_fp, "rb")
        i = 0
        while i < sample_size:
            try:
                data_list.append(pickle.load(pklfl))
                print(len(data_list))
                i += 1
            except EOFError:
                break

        pklfl.close()
        data_list = pd.DataFrame(
            data_list, columns=["Genre", "Name", "Scene", "Fp", "Data"]
        )
        if self.supervised:
            data_set = TransVidDataset(data_list)
        else:
            data_set = ContrastiveDataSet(data_list)

        return data_set


# For unsupervised training
class ContrastiveDataSet(Dataset):
    def __init__(self, data_frame):
        super(Dataset, self).__init__()
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fp = self.data_frame.at[idx, "Fp"]
        name = self.data_frame.at[idx, "Name"]
        scene = self.data_frame.at[idx, "Scene"]
        genre = self.data_frame.at[idx, "Genre"]
        data = self.data_frame.at[idx, "Data"]
        chunk_zi = data[0].transpose(1, 0, 2, 3)
        chunk_zj = data[random.randrange(1, len(data))].transpose(1, 0, 2, 3)
        sample = {
            "fp": fp,
            "name": name,
            "scene": scene,
            "genre": genre,
            "data": (chunk_zi, chunk_zj),
        }
        return sample
