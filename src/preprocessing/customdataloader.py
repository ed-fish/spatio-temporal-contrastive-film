from __future__ import print_function, division
from torch.utils.data import Dataset
import warnings
import random
import pickle
import torch
import pandas as pd

GENRE = "Genre"
FILEPATH = "Filepath"
SCENE = "Scene"
T_DATA = "Transformed_Data"
O_DATA = "Original_Data"
NAME = "Name"


class DataLoader:
    def __init__(self, data_fp, sample_size, supervised=False):
        self.supervised = supervised
        self.data_frame = self.__load_data__(data_fp, sample_size)

    def __load_data__(self, data_fp, sample_size):
        data_list = []
        pklfl = open(data_fp, "rb")
        i = 0
        while i < sample_size:
            try:
                data_list.append(pickle.load(pklfl))

                i += 1
            except EOFError:
                break

        pklfl.close()
        data_list = pd.DataFrame(
            data_list,
            columns=[GENRE, NAME, FILEPATH, SCENE, T_DATA],
        )

        return data_list

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fp = self.data_frame.at[idx, FILEPATH]
        name = self.data_frame.at[idx, NAME]
        scene = self.data_frame.at[idx, SCENE]
        genre = self.data_frame.at[idx, GENRE]
        t_data = self.data_frame.at[idx, T_DATA]
        random_n = random.randrange(0, len(t_data))

        chunk_zi = t_data[random_n]
        del t_data[random_n]
        chunk_zj = t_data[random.randrange(0, len(t_data))]
        image_zi = chunk_zi[1]
        image_zj = chunk_zj[0]
        chunk_zi = chunk_zi.transpose(1, 0, 2, 3)
        chunk_zj = chunk_zj.transpose(1, 0, 2, 3)
        chunk_zi = torch.FloatTensor(chunk_zi)
        chunk_zj = torch.FloatTensor(chunk_zj)
        t_pair = [chunk_zi, chunk_zj, image_zi, image_zj]
        sample = {
            FILEPATH: fp,
            NAME: name,
            SCENE: scene,
            GENRE: genre,
            T_DATA: t_pair,
        }
        return sample
