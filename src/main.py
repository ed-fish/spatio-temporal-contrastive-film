from src.preprocessing import dataprocessing as dp
import pandas as pd
from src.preprocessing.customdataloader import TransVidDataset
import torch
from torchvision import models
import os
import shutil
from sklearn.cluster import KMeans
import torchvision
import numpy

data = "/home/ed/PhD/Temporal-3DCNN-pytorch/data/debug/data30k.pkl"

def data_loading(data_frame, train_len, bs):
    transform = torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))
    data_frame = data_frame.sample(frac=1)
    dataset = TransVidDataset(data_frame)
    train, test = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle= True, num_workers = 4, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = bs, shuffle = True, num_workers = 4, drop_last = True)
    return train_loader, test_loader

def create_data_frame(text_file, out_pkl):
    data_frame = dp.create_data_frame(text_file, debug=True)
    transform_df = dp.create_trans_data_frame(data_frame)
    transform_df.to_pickle(out_pkl)

def read_data_frame(pkl_file):
    return pd.read_pickle(pkl_file)

def load_model():
    resnet = models.video.r3d_18(pretrained=True)
    return resnet

def train_model(model, train, test, epochs, learning_rate, gpu=False):

    # optimization as described in paper
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    device = None
    if gpu:
        device = torch.device("cuda:0")
    modes = ["train", "val"]
    epoch = 0
    while epoch < epochs:
        for state in modes:
            running_loss = 0.0
            total = 0.0

            for i, batch in enumerate((train, test)):
                model.train() if state == 'train' else model.eval()
                optimizer.zero_grad()
                labels = batch['scene']
                data = batch['data']

                if gpu: labels = labels.to(device)
                embedding, outputs = model(data)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += outputs.size(0)

                if state == 'train':
                    loss.backward()
                    optimizer.step()

def main():
    print("processing data")
    df = read_data_frame(data)
    train_set, test_set = data_loading(df, 500, 1)
    model = load_model()
    model = model.eval()
    outputs = []
    file_paths = []
    names = []

    with torch.no_grad():
        for i in train_set:
            output = model(i['data'].permute(0,2,1,3,4).float())
            vid = i['fp']
            label = i['genre']
            name = i['name']
            print(vid)
            outputs.append(output.numpy().squeeze(0))
            file_paths.append(vid)
            names.append(name)

    k = 100
    kmodel = KMeans(n_clusters= k, n_jobs=4, random_state=728)
    kmodel.fit(outputs)
    kpredictions = kmodel.predict(outputs)
    for i in range(k):
        os.makedirs(os.path.join("/home/ed/PhD/Temporal-3DCNN-pytorch/tests/",str(i)), exist_ok=True)
    for i in range(len(outputs)):
        outpath = "".join(file_paths[i][0].split("/")[5:7]) + ".mp4"
        shutil.copy(file_paths[i][0], "/home/ed/PhD/Temporal-3DCNN-pytorch/tests/" + str(kpredictions[i]) + '/'+ outpath )







if __name__ == "__main__":
    main()