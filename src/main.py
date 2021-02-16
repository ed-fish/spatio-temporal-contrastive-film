# Viz
from torch.utils.tensorboard import SummaryWriter

# Pytorch
import torch

# Custom
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from preprocessing.customdataloader import (
    TransVidDataset,
    ContrastiveDataSet,
    DataLoader,
)
from preprocessing import dataprocessing as dd


logging_object = {
    "writer": SummaryWriter(),
    "directory": "/home/ed/PhD/Temporal-3DCNN-pytorch/logs/",
    "interval": 5,
}


def main(train=False):
    spatio_model = SpatioTemporalContrastiveModel(512, 128)
    data_set = DataLoader(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/data2500.pkl", 2500
    )
    if train:
        loss = NT_Xent(32, 0.5, 1)
        spatio_model.train(
            data_set,
            32,
            torch.optim.Adam(spatio_model.parameters(), 0.08),
            300,
            logging_object,
            loss,
            True,
        )
    else:
        weights = "/home/ed/PhD/Temporal-3DCNN-pytorch/model165.pt"
        spatio_model.eval_model(data_set, 1, logging_object, weights, True, debug=True)


if __name__ == "__main__":
    main()
