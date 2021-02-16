# Sys
import os

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
from eval.visualisation import Visualisation
from preprocessing import dataprocessing as dd


class Config:
    def __init__(
        self,
        learning_rate,
        batch_size,
        base_directory,
        sample_size,
        input_layer_size,
        output_layer_size,
        epochs,
        gpu=True,
    ):
        run_directory = os.path.join(
            base_directory, str(sample_size), str(learning_rate)
        )
        self.feature_directory = os.path.join(run_directory, "features")
        self.eval_directory = os.path.join(run_directory, "eval")
        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(self.eval_directory, exist_ok=True)
        os.makedirs(self.feature_directory, exist_ok=True)
        self.writer = SummaryWriter(run_directory)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.epochs = epochs
        self.gpu = gpu


logging = Config(
    0.5, 32, "/home/ed/PhD/Temporal-3DCNN-pytorch/logs", 100, 512, 128, 10, True
)


def main(input_data, config, train=False):
    spatio_model = SpatioTemporalContrastiveModel(
        config.input_layer_size, config.output_layer_size
    )
    data_set = DataLoader(input_data, config.sample_size)
    if train:
        loss = NT_Xent(config.batch_size, 0.5, 1)
        spatio_model.train(
            data_set,
            torch.optim.Adam(spatio_model.parameters(), config.learning_rate),
            loss,
            config,
        )
    else:
        weights = config.feature_directory + "/model.pt"
        spatio_model.eval_model(
            data_set,
            config,
            weights,
        )
        vis = Visualisation(config)
        vis.tsne()


if __name__ == "__main__":
    main(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/data2500.pkl",
        logging,
        False,
    )
