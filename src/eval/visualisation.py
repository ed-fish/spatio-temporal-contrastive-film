import pickle as pkl
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Visualisation:
    def __init__(self, data_frame, logging_object):
        self.data_frame = pd.read_pickle(data_frame)
        self.writer = logging_object["writer"]

    def tsne(self):
        names = self.data_frame["name"].tolist()
        data = self.data_frame["data"].tolist()
        data = np.stack(data)
        print(data.shape)
        data = data.squeeze(1)
        self.writer.add_embedding(data, names)


logging_object = {
    "writer": SummaryWriter(),
    "directory": "/home/ed/PhD/Temporal-3DCNN-pytorch/logs",
    "interval": 5,
}
viz = Visualisation(
    "/home/ed/PhD/Temporal-3DCNN-pytorch/logs/output.pkl", logging_object
)
viz.tsne()


def pca_select_reduce(outputs):
    X_Std = StandardScaler().fit_transform(outputs)
    pca = PCA(n_components=15)
    principal_components = pca.fit_transform(X_Std)
    pca_components = pd.DataFrame(principal_components)
    plt.scatter(pca_components[0], pca_components[1], alpha=0.1, color="black")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()
    return pca_components


def find_k(pca_components):
    K = range(1, 50)
    sum_sq_dis = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(pca_components.iloc[:, :3])
        sum_sq_dis.append(km.inertia_)
    plt.plot(K, sum_sq_dis, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.show()
