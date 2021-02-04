from preprocessing import dataprocessing as dp
import pandas as pd
from preprocessing.customdataloader import TransVidDataset, ContrastiveDataSet
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from torchvision import models
import torch
import torch.nn as nn
import os
import shutil
from sklearn.cluster import KMeans

def data_loading(data_frame, train_len, bs):
    data_frame = data_frame.sample(frac=1)
    dataset = ContrastiveDataSet(data_frame)
    print('dataset length', len(dataset))
    train, test = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle= True, num_workers = 4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = bs, shuffle = True, num_workers = 4, drop_last=True)
    return train_loader, test_loader

def read_data_frame(pkl_file):
    return pd.read_pickle(pkl_file)

def load_model():
    resnet = models.video.r3d_18(pretrained=True)
    return resnet

def train_model(model, train, test, batch_size, epochs, learning_rate, gpu=False):
    # optimization as described in paper
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    device = None
    if gpu:
        device = torch.device("cuda:0")
    contrastive_loss = NT_Xent(batch_size, 0.5, 1)
    model = nn.DataParallel(model)
    model = model.to(device)
    modes = ["train", "val"]
    epoch = 0
    model.train()
    
    while epoch < epochs:
        running_loss = 0.0
        total = 0.0

        for batch in train:
            optimizer.zero_grad()
            data = batch['data']
            zi = data[0].to(device)
            zj = data[1].to(device)
            zi_embedding = model(zi)
            zj_embedding = model(zj)
            loss = contrastive_loss.forward(zi_embedding, zj_embedding)
            print(loss.item())
            loss.backward()
            optimizer.step()


                
def kmeans(outputs, file_paths, k=5):
    """ Completes KMEANS clustering and saves
        images to test directory within their clusters."""
    kmodel = KMeans(n_clusters=k, n_jobs=4, random_state=728)
    kmodel.fit(outputs)
    kpredictions = kmodel.predict(outputs)

    for i in range(k):
        os.makedirs(os.path.join("/home/ed/PhD/Temporal-3DCNN-pytorch/tests/", str(i)), exist_ok=True) #todo add to arg parser
    for i in range(len(outputs)):
        outpath = "".join(file_paths[i][0].split("/")[5:7]) + ".mp4"
        shutil.copy(file_paths[i][0],
                    "/home/ed/PhD/Temporal-3DCNN-pytorch/tests/" + str(kpredictions[i]) + '/' + outpath)


def eval_model(df, sample_size, batch_size):
    train_set, test_set = data_loading(df, sample_size, batch_size)
    model = load_model()
    model = model.eval()
    outputs = []
    file_paths = []

    with torch.no_grad():
        for i in train_set:
            data = i['data'][0].permute(0,3,2,1,4)
            output = model(data.float())
            vid = i['fp']
            print(vid)
            outputs.append(output.numpy().squeeze(0))
            file_paths.append(vid)
            print(len(file_paths))
            print(len(outputs))
    kmeans(outputs, file_paths, k=20)

def pre_process_data(samples):
    df = dp.create_data_frame("/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/original/cache-file-paths.txt")
    tr = dp.create_trans_data_frame(df, samples)
    tr.to_pickle("data.pkl")

def main():
    device = torch.device("cuda:0")
    model = SpatioTemporalContrastiveModel()
    model = model.add_projector()
    batch_size = 50
    df = read_data_frame("/home/ed/PhD/Temporal-3DCNN-pytorch/src/data.pkl")
    train_set, test_set = data_loading(df, 50, batch_size)
    train_model(model, train_set, test_set, batch_size, 10, 0.001, True)

if __name__ == "__main__":
    main()
