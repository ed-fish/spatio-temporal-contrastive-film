# Viz
import matplotlib as plt
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Pytorch
import torch
import torch.nn as nn
from torchvision import models

# Std
import os
import shutil

# Data
import pandas as pd
import pickle

# Custom
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from preprocessing.customdataloader import (
    TransVidDataset,
    ContrastiveDataSet,
    DataLoader,
)
from preprocessing import dataprocessing as dd

# SKData
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# fix me
def read_data_frame(pkl_file):
    return pd.read_pickle(pkl_file)


# refactor me please
def kmeans(outputs, file_paths, k=250):
    """Completes KMEANS clustering and saves
    images to test directory within their clusters."""
    kmodel = KMeans(n_clusters=k, n_jobs=4, random_state=728)
    kmodel.fit(outputs)
    kpredictions = kmodel.predict(outputs)

    for i in range(k):
        os.makedirs(
            os.path.join("/home/ed/PhD/Temporal-3DCNN-pytorch/tests/", str(i)),
            exist_ok=True,
        )  # todo add to arg parser
    for i in range(len(outputs)):
        outpath = "".join(file_paths[i][0].split("/")[5:7]) + ".mp4"
        shutil.copy(
            file_paths[i][0],
            "/home/ed/PhD/Temporal-3DCNN-pytorch/tests/"
            + str(kpredictions[i])
            + "/"
            + outpath,
        )


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


def eval_model(model_pth, df, batch_size):
    data_set = data_loading(df, batch_size)
    model = SpatioTemporalContrastiveModel(pretrained=True)
    model = model.add_projector()
    model.load_state_dict(torch.load(model_pth))
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)
    model = model.to(torch.device("cuda:0"))
    outputs = []
    file_paths = []

    with torch.no_grad():
        for i in data_set:
            data = i["data"][0].to(torch.device("cuda:0"))
            output = model(data.float())
            output = output.cpu()
            vid = i["fp"]
            print(output.shape)
            outputs.append(output.numpy().squeeze((0, 2, 3, 4)))
            file_paths.append(vid)
            print(len(file_paths))
            print(len(outputs))
    kmeans(outputs, file_paths)


def pre_process_data(samples, fp):
    df = dp.create_data_frame(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/original/cache-file-paths.txt"
    )
    tr = dp.create_trans_data_frame(df, samples, fp)
    df = dd.from_pandas(tr)
    tr.to_pickle("data{}.pkl".format(samples))


logging_object = {
    "writer": SummaryWriter(),
    "directory": "/home/ed/PhD/Temporal-3DCNN-pytorch/logs",
    "interval": 5,
}


def main():
    spatio_model = SpatioTemporalContrastiveModel(512, 128)
    data_set = DataLoader(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/data2500.pkl", 100
    )
    spatio_model.train(
        data_set,
        32,
        torch.optim.Adam(spatio_model.parameters(), 0.5),
        10,
        logging_object,
        NT_Xent,
        True,
    )


if __name__ == "__main__":
    main()
