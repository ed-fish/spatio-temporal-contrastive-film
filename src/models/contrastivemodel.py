from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle as pkl


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# See Tomcat, B. Twitch 2021
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def train(
        self,
        data_loader,
        optimizer,
        loss_alg,
        config,
    ):
        if config.gpu:
            device = torch.device("cuda:0")
            self.to(device)

        epoch = 0
        best_loss = 100.0
        best_epoch = 0
        data_loader = torch.utils.data.DataLoader(
            data_loader, config.batch_size, shuffle=True, num_workers=6, drop_last=True
        )
        while epoch < config.epochs:
            total = 0
            running_loss = 0
            for bn, batch in enumerate(data_loader):
                optimizer.zero_grad()
                data = batch["data"]
                zi = data[0].to(device)
                zj = data[1].to(device)
                zi_embedding = self.forward(zi)
                zj_embedding = self.forward(zj)
                loss = loss_alg.forward(zi_embedding, zj_embedding)
                running_loss += loss.item()
                total += config.batch_size
                loss.backward()
                optimizer.step()
                if bn % 10:
                    print("batch n", bn, loss)
            if running_loss < best_loss:
                torch.save(self.state_dict(), config.feature_directory + "/model.pt")
                best_loss = running_loss
                best_epoch = epoch

            print(f"Epoch {epoch} \n Loss : {running_loss/total}")
            config.writer.add_scalar("training loss", running_loss / total, epoch)
            config.writer.flush()
            epoch += 1

        print(
            "Model completed training. Final loss = {}, Best loss = {}, Best Epoch = {}".format(
                running_loss, best_loss, best_epoch
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
            data_loader, 1, shuffle=True, num_workers=6, drop_last=True
        )
        self.load_state_dict(torch.load(model_features), strict=False)
        self.base_model.fc = Identity()
        self = self.to(torch.device("cuda:0"))
        print(self)

        output_df = []
        with torch.no_grad():
            for i in data_loader:
                data = i["data"][0].to(torch.device("cuda:0"))
                output = self(data.float())
                output = output.cpu()
                if debug:
                    print("outputs shape from model", output.shape)
                output = output.numpy().squeeze(0)
                output_df.append([i["name"], i["fp"], i["scene"], output])
        output_df = pd.DataFrame(output_df, columns=["name", "fp", "scene", "data"])
        filepath = config.eval_directory + "/eval_output.pkl"
        output_df.to_pickle(filepath)
        print(output_df)


class SpatioTemporalContrastiveModel(Model):
    def __init__(self, input_layer_size, output_layer_size, pretrained=False):
        super(SpatioTemporalContrastiveModel, self).__init__()
        if pretrained:
            self.base_model = models.video.r3d_18(pretrained=True)
        else:
            self.base_model = models.video.r3d_18(pretrained=False)

        self.base_model.fc = nn.Sequential(
            nn.Linear(512, input_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_layer_size, output_layer_size),
        )

        self.create_model()

    def create_model(self):
        return self.base_model

    def print_model(self):
        print(self.base_model)

    def forward(self, x):
        output = self.base_model(x)
        return output


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
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

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