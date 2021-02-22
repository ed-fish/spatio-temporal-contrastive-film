# Sys
import os

# Viz
from torch.utils.tensorboard import SummaryWriter

# Pytorch
import torch

# Custom
from models.contrastivemodel import SpatioTemporalContrastiveModel, NT_Xent
from preprocessing.customdataloader import (
    DataLoader,
)
from eval.visualisation import Visualisation
from preprocessing.dataprocessing import DataTransformer

""" An implementation of Spatio-Temporal Constrastive Networks for Video
    https://arxiv.org/pdf/2008.03800.pdf"""


class Config:
    """ Loads a configuration for the model"""

    def __init__(
        self,
        learning_rate,
        batch_size,
        base_directory,
        trans_data_dir,
        cache_file,
        sample_size,
        input_layer_size,
        output_layer_size,
        epochs,
        n_frozen_layers,
        mean,
        std,
        gpu=True,
    ):
        run_directory = os.path.join(
            base_directory, str(sample_size), str(learning_rate)
        )
        self.feature_directory = os.path.join(run_directory, "features")
        self.eval_directory = os.path.join(run_directory, "eval")
        self.trans_data_dir = trans_data_dir
        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(self.eval_directory, exist_ok=True)
        os.makedirs(self.feature_directory, exist_ok=True)
        os.makedirs(self.trans_data_dir, exist_ok=True)
        self.cache_file = cache_file
        self.writer = SummaryWriter(run_directory)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.epochs = epochs
        self.gpu = gpu
        self.mean = mean
        self.std = std
        self.n_frozen_layers = n_frozen_layers


# Setup logging object
logging1 = Config(
    learning_rate=0.16,
    batch_size=30,
    base_directory="/home/ed/PhD/Temporal-3DCNN-pytorch/logs/layers_14/",
    trans_data_dir="/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed",
    cache_file="/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/original/cache-file-paths.txt",
    sample_size=1400,
    input_layer_size=512,  # Projection head 1 g0
    output_layer_size=128,  # Projection head 2 h0
    epochs=100,
    n_frozen_layers=14,
    mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989),
    gpu=True,  # Currently no cpu support
)
logging2 = Config(
    learning_rate=0.16,
    batch_size=30,
    base_directory="/home/ed/PhD/Temporal-3DCNN-pytorch/logs/",
    trans_data_dir="/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed",
    cache_file="/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/original/cache-file-paths.txt",
    sample_size=3000,
    input_layer_size=512,  # Projection head 1 g0
    output_layer_size=128,  # Projection head 2 h0
    epochs=100,
    n_frozen_layers=45,
    mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989),
    gpu=True,  # Currently no cpu support
)


def main(input_data, config, train=False):

    # load model with config
    spatio_model = SpatioTemporalContrastiveModel(
        config.input_layer_size, config.output_layer_size, pretrained=True
    )

    # load dataset sample size = 2(n-1)
    data_set = DataLoader(input_data, config.sample_size)

    if train:
        loss = NT_Xent(
            config.batch_size, 0.5, 1
        )  # temperature and world size preset as per paper
        spatio_model.train_model(
            data_set,
            torch.optim.Adam(
                filter(lambda p: p.requires_grad, spatio_model.parameters()),
                config.learning_rate,
            ),
            loss,
            config,
        )
    else:
        weights = config.feature_directory + "/model.pt"
        print(weights)
        spatio_model.eval_model(
            data_set,
            config,
            weights,
        )
        vis = Visualisation(config)
        # Creates tensorboard t-sne plot in config.rundirectory
        vis.tsne()
        vis.kmeans(200)


def data_creation(logger):
    data_transformer = DataTransformer(logger)
    data_transformer.transform_data_from_cache()


if __name__ == "__main__":

    data_creation(logging2)

    """main(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/4500.pkl",
        logging1,
        True,
    )

    main(
        "/home/ed/PhD/Temporal-3DCNN-pytorch/data/input/transformed/4500.pkl",
        logging2,
        True,
    )"""
