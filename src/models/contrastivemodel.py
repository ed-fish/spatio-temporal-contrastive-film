from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle as pkl
from preprocessing.customdataloader import GENRE, NAME, SCENE, FILEPATH, O_DATA, T_DATA


DEVICE = "cuda:0"


class Identity(nn.Module):
    """ Identity module used to remove projection head after training"""

    def forward(self, x):
        return x


class SpatioTemporalContrastiveModel(nn.Module):
    def __init__(self):
        super(SpatioTemporalContrastiveModel, self).__init__()
        self.input_layer_size = 512
        self.output_layer_size = 128

        self.video_model = models.video.r3d_18(pretrained=True)
        self.image_model = models.resnet18(pretrained=True)

        self.video_model.fc = nn.Sequential(
            nn.Linear(512, self.input_layer_size),
        )
        self.image_model.fc = nn.Sequential(
            nn.Linear(512, self.input_layer_size),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.input_layer_size * 2, self.input_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_layer_size, self.output_layer_size),
        )

    def print_model(self):
        print("VIDEO MODEL")
        print(self.video_model)
        print("IMAGE MODEL")
        print(self.image_model)

    def forward_model(self, x, y, train=True):
        output_video = self.video_model(x)

        output_image = self.image_model(y)
        output = torch.cat((output_video, output_image), -1)
        if train:
            output = self.projection_head(output)
        return output

    def train_model(
        self,
        data_loader,
        optimizer,
        loss_alg,
        config,
    ):
        print("training")
        self.video_model.train(True)
        self.image_model.train(True)
        if config.n_frozen_layers != 0:
            c = 0
            for name, param in self.video_model.named_parameters():
                if c <= config.n_frozen_layers:
                    param.requires_grad = False
                    print(c, name, param.requires_grad)
                    c += 1
            for name, param in self.video_model.named_parameters():
                if c <= config.n_frozen_layers:
                    param.requires_grad = False
                    print(c, name, param.requires_grad)
                    c += 1

        if config.gpu:
            device = torch.device(DEVICE)
            self.video_model.to(device)
            self.image_model.to(device)
            self.projection_head.to(device)

        epoch = 0
        best_loss = 2000.0
        best_epoch = 0
        print("loading data")
        data_loader = torch.utils.data.DataLoader(
            data_loader,
            config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        print("dataloaded")

        while epoch < config.epochs:
            total = 0
            running_loss = 0
            for bn, batch in enumerate(data_loader):
                optimizer.zero_grad()
                t_data = batch[T_DATA]
                zi_v = t_data[0].to(device)  # Augmented sample : 1 e (x ...xn) e v(i)
                zj_v = t_data[1].to(device)  # Augmented sample : 2 e (x ...xn) e v(i)
                zi_i = t_data[2].to(device)
                zj_i = t_data[3].to(device)
              
                zi_embedding = self.forward_model(zi_v, zi_i)
                zj_embedding = self.forward_model(zj_v, zj_i)
                loss = loss_alg.forward(zi_embedding, zj_embedding)
                running_loss += loss.item()
                total += config.batch_size
                loss.backward()
                optimizer.step()
                if bn % 5 == 0:
                    print("batch n", bn, loss)
            print(f"running loss {running_loss}, best_loss{best_loss}")
            if running_loss < best_loss:
                print("model should save")
                out_dir = os.path.join(config.feature_directory, "model.pt")
                print(out_dir)
                torch.save(self.state_dict(), config.feature_directory + "/model.pt")
                best_loss = running_loss
                best_epoch = epoch

            print(f"Epoch {epoch} \n Loss : {running_loss/total}")
            config.writer.add_scalar("training loss", running_loss / total, epoch)
            for name, weight in self.named_parameters():
                config.writer.add_histogram(name, weight, epoch)

            config.writer.flush()
            epoch += 1

        print(
            "Model completed training. Final loss = {}, Best loss = {}, Best Epoch = {}".format(
                running_loss / total, best_loss / total, best_epoch
            )
        )

    def eval_model(
        self,
        data_loader,
        config,
        model_features,
        debug=False,
    ):
        data_loader = torch.utils.data.DataLoader(
            data_loader,
            1,
            shuffle=True,
            num_workers=6,
            drop_last=True,
        )
        # self.load_state_dict(torch.load(model_features), strict=False)
        self.video_model.fc = Identity()
        self.image_model.fc = Identity()
        self.video_model.to(torch.device(DEVICE))
        self.image_model.to(torch.device(DEVICE))

        self.video_model.eval()
        self.image_model.eval()

        output_df = []
        with torch.no_grad():
            for i in data_loader:
                v_data1 = i[T_DATA][0].to(torch.device(DEVICE))
                i_data1 = i[T_DATA][2].to(torch.device(DEVICE))

                output1 = self.forward_model(v_data1, i_data1, train=False)
                output1 = output1.cpu()

                if debug:
                    print("outputs shape from model", output1.shape)
                output1 = output1.numpy().squeeze(0)
                output_df.append(
                    [
                        i[NAME],
                        i[FILEPATH],
                        i[SCENE],
                        i[GENRE],
                        output1,
                        i[T_DATA][2].cpu(),
                    ]
                )

        output_df = pd.DataFrame(
            output_df, columns=[NAME, FILEPATH, SCENE, GENRE, T_DATA, "Image"]
        )
        filepath = config.eval_directory + "/eval_output.pkl"
        output_df.to_pickle(filepath)
        print(output_df)


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        # this is a comment
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
