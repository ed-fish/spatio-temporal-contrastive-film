from torchvision import models
import torch
import torch.nn as nn
import numpy as np

# See Tomcat, B. Twitch 2021
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def train(
        self,
        data_loader,
        batch_size,
        optimizer,
        epochs,
        logging_object,
        loss_alg,
        gpu=False,
    ):
        if gpu:
            device = torch.device("cuda:0")
            self.to(device)

        epoch = 0

        best_loss = 5
        best_epoch = 0
        data_loader = torch.utils.data.DataLoader(
            data_loader, batch_size, shuffle=True, num_workers=6, drop_last=True
        )
        while epoch < epochs:
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
                total += batch_size
                loss.backward()
                optimizer.step()
                if bn % logging_object["interval"]:
                    print()

            if running_loss < best_loss:
                torch.save(
                    self.state_dict(),
                    logging_object["directory"] + "/model{}.pt".format(epoch),
                )
                best_loss = running_loss
                best_epoch = epoch

            print("Epoch {} \n Loss : {}".format(epoch, running_loss / total))
            logging_object["writer"].add_scalar(
                "training loss", running_loss / total, epoch
            )

            logging_object["writer"].flush()
            epoch += 1

        print(
            "Model completed training. Final loss = {}, Best loss = {}, Best Epoch = {}".format(
                running_loss, best_loss, best_epoch
            )
        )


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