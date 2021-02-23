import pickle as pkl
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from sklearn.cluster import KMeans
import os
import shutil
from preprocessing.dataprocessing import FILEPATH, NAME, SCENE, T_DATA, O_DATA


class Visualisation:
    def __init__(self, config, kmeans=True, clusters=10):
        data_frame = pd.read_pickle(config.eval_directory + "/eval_output.pkl")
        self.writer = config.writer
        self.cluster_dir = os.path.join(config.eval_directory, "clusters")
        self.data = data_frame[T_DATA].tolist()
        self.names = data_frame[NAME].tolist()
        self.file_paths = data_frame[FILEPATH].tolist()
        self.scenes = data_frame[SCENE].tolist()
        self.images = data_frame["Image"].tolist()

    def tsne(self):
        data = np.stack(self.data)
        images = torch.stack(self.images)
        images = images.squeeze(1)
        print("image shape!", images.shape)
        print(data.shape)

        self.writer.add_embedding(data, label_img=images)

    def kmeans(self, clusters):
        """Completes KMEANS clustering and saves
        images to test directory within their clusters."""
        kmodel = KMeans(n_clusters=clusters, n_jobs=4, random_state=728)
        outputs = self.data
        kmodel.fit(outputs)
        kpredictions = kmodel.predict(outputs)
        os.makedirs(self.cluster_dir, exist_ok=True)

        for i in range(clusters):
            os.makedirs(
                os.path.join(self.cluster_dir, str(i)),
                exist_ok=True,
            )  # todo add to arg parser
        for i in range(len(outputs)):
            outpath = (
                "".join(self.file_paths[i][0].split("/")[5:7]) + ".mp4"
            )  # todo this only works with my file pattern
            shutil.copy(
                self.file_paths[i][0],
                self.cluster_dir + "/" + str(kpredictions[i]) + "/" + outpath,
            )
