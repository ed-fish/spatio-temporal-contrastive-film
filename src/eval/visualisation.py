import pickle as pkl
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil


class Visualisation:
    def __init__(self, data_frame, logging_object):
        data_frame = pd.read_pickle(data_frame)
        self.writer = logging_object["writer"]
        self.data = data_frame["data"].tolist()
        self.names = data_frame["name"].tolist()
        self.file_paths = data_frame["fp"].tolist()
        self.scenes = data_frame["scene"].tolist()

    def tsne(self):
        data = np.stack(self.data)
        print(data.shape)
        data = data.squeeze(1)
        self.writer.add_embedding(data, self.names)

    def kmeans(self, k=400):
        """Completes KMEANS clustering and saves
        images to test directory within their clusters."""
        kmodel = KMeans(n_clusters=k, n_jobs=4, random_state=728)
        outputs = self.data
        kmodel.fit(outputs)
        kpredictions = kmodel.predict(outputs)

        for i in range(k):
            os.makedirs(
                os.path.join("/home/ed/PhD/Temporal-3DCNN-pytorch/tests/", str(i)),
                exist_ok=True,
            )  # todo add to arg parser
        for i in range(len(outputs)):
            outpath = "".join(self.file_paths[i][0].split("/")[5:7]) + ".mp4"
            shutil.copy(
                self.file_paths[i][0],
                "/home/ed/PhD/Temporal-3DCNN-pytorch/tests/"
                + str(kpredictions[i])
                + "/"
                + outpath,
            )


logging_object = {
    "writer": SummaryWriter(),
    "directory": "/home/ed/PhD/Temporal-3DCNN-pytorch/logs",
    "interval": 5,
}
viz = Visualisation(
    "/home/ed/PhD/Temporal-3DCNN-pytorch/src/output.pkl", logging_object
)
viz.kmeans()
