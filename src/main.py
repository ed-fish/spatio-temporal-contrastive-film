import torch
import torch.nn as nn
import os
import shutil
import matplotlib as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from preprocessing import dataprocessing as dp
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pickle

from preprocessing.customdataloader import TransVidDataset, ContrastiveDataSet
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from torchvision import models

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def data_loading(data_frame, train_len, bs):
    data_frame = data_frame.sample(frac=1)
    dataset = ContrastiveDataSet(data_frame)
    print("dataset length", len(dataset))
    train, test = torch.utils.data.random_split(
        dataset, [train_len, len(dataset) - train_len]
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=bs, shuffle=True, num_workers=4, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=bs, shuffle=True, num_workers=4, drop_last=True
    )
    return train_loader, test_loader


def read_data_frame(pkl_file):
    return pd.read_pickle(pkl_file)


def load_model():
    resnet = models.video.r3d_18(pretrained=True)
    return resnet


def train_model(model, train, test, batch_size, epochs, learning_rate):

    logging_dir = os.path.join("logs", str(learning_rate))
    os.makedirs(logging_dir, exist_ok=True)
    print(logging_dir)
    model_dir = os.path.join(logging_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir="logs/" + str(learning_rate)
    )  # todo convert to argparser
    # optimization as described in paper
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    contrastive_loss = NT_Xent(batch_size, 0.5, 1)
    device = torch.device("cuda:0")
    states = ["train", "val"]
    epoch = 0
    best_loss = 100.0
    while epoch < epochs:
        for state in states:

            total = 0.0
            running_loss = 0.0
            data_set = train if state == "train" else test
            model.train() if state == "train" else model.eval()
            for _, batch in enumerate(data_set):
                optimizer.zero_grad()
                data = batch["data"]
                zi = data[0].to(device)
                zj = data[1].to(device)
                zi_embedding = model(zi)
                zj_embedding = model(zj)
                loss = contrastive_loss.forward(zi_embedding, zj_embedding)
                running_loss += loss.item()
                total += zi_embedding.size(0)

                if state == "train":
                    loss.backward()
                    optimizer.step()
                else:
                    if running_loss < best_loss:
                        torch.save(model.state_dict(), logging_dir + "/model.pt")
                        best_loss = running_loss

            print("{} \n Epoch {} \n Loss : {}".format(state, epoch, loss / total))
            writer.add_scalar("Loss/{}".format(state), running_loss, epoch)
            writer.flush()

        epoch += 1


def kmeans(outputs, file_paths, k=5):
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


def find_k(outputs):
    K = range(1, 100)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(outputs)
        sum_sq_dis.append(km.inertia_)
        plt.plot(K, Sum_of_squared_distances, "bx-")
        plt.xlabel("k")
        plt.ylabel("Sum_of_squared_distances")
        plt.title("Elbow Method For Optimal k")
        plt.show()


def eval_model(df, sample_size, batch_size):
    train_set, test_set = data_loading(df, sample_size, batch_size)
    model = load_model()
    model = model.eval()
    outputs = []
    file_paths = []

    with torch.no_grad():
        for i in test_set:
            data = i["data"][0].permute(0, 3, 2, 1, 4)
            output = model(data.float())
            vid = i["fp"]
            print(vid)
            outputs.append(output.numpy().squeeze(0))
            file_paths.append(vid)
            print(len(file_paths))
            print(len(outputs))
    kmeans(outputs, file_paths, k=20)


def pre_process_data(samples, fp):
    df = dp.create_data_frame(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/original/cache-file-paths.txt"
    )
    tr = dp.create_trans_data_frame(df, samples, fp)
    df = dd.from_pandas(tr)
    tr.to_pickle("data10k.pkl")


def load_data(fp):
    data_list = []
    pklfl = open(fp, "rb")
    i = 0
    while i < 2500:
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
    return data_list


def main():
    torch.manual_seed(0)
    learning_rate = 0.4
    device = torch.device("cuda:0")
    model = SpatioTemporalContrastiveModel()
    model = model.add_projector()
    model = model.to(device)
    batch_size = 16
    df = load_data(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/data2500.pkl"
    )
    train_set, test_set = data_loading(df, 2000, batch_size)
    train_model(
        model, train_set, test_set, batch_size, epochs=100, learning_rate=learning_rate
    )

    # pre_process_data(0, "pickle_test.pkl")


if __name__ == "__main__":
    main()
