import pandas as pd
import numpy as np
import torch
import csv
from sklearn.cluster import KMeans
from torchvision.utils import save_image
import os
import shutil
from preprocessing.customdataloader import FILEPATH, NAME, SCENE, T_DATA, GENRE


class Visualisation:
    def __init__(self, config, kmeans=True, plt=True, clusters=10):
        self.data_frame = pd.read_pickle(config.eval_directory + "/eval_output.pkl")
        self.writer = config.writer
        self.cluster_dir = os.path.join(config.eval_directory, "clusters")
        self.plt_dir = os.path.join(config.eval_directory, "plt")
        self.data = self.data_frame[T_DATA].tolist()
        self.names = self.data_frame[NAME].tolist()
        self.file_paths = self.data_frame[FILEPATH].tolist()
        self.scenes = self.data_frame[SCENE].tolist()
        self.images = self.data_frame["Image"].tolist()
        self.genre = self.data_frame[GENRE].tolist()
        self.config = config
        self.reverse_norm()

    def inverse_norm(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        return tensor

    def reverse_norm(self):
        for x, i in enumerate(self.images):
            t = self.inverse_norm(i, self.config.mean, self.config.std)
            self.images[x] = t

    def tsne(self):
        data = np.stack(self.data)
        images = torch.stack(self.images)
        images = images.squeeze(1)
        print("image shape!", images.shape)
        print(data.shape)
        self.writer.add_embedding(data, label_img=images)

    def kmeans(self, clusters):
        """Completes KMEANS clustering and saves
        images to test directory within their clusters."""
        kmodel = KMeans(n_clusters=clusters, n_jobs=4, random_state=728)
        outputs = self.data
        kmodel.fit(outputs)
        kpredictions = kmodel.predict(outputs)
        os.makedirs(self.cluster_dir, exist_ok=True)

        for i in range(clusters):
            os.makedirs(
                os.path.join(self.cluster_dir, str(i)),
                exist_ok=True,
            )  # todo add to arg parser
        for i in range(len(outputs)):
            outpath = (
                "".join(self.file_paths[i][0].split("/")[5:7]) + ".mp4"
            )  # todo this only works with my file pattern
            shutil.copy(
                self.file_paths[i][0],
                self.cluster_dir + "/" + str(kpredictions[i]) + "/" + outpath,
            )

    # FILENAME, GENRE, NAME, YEAR
    # FNS, G1|G2|G3|, NAME, YEAR

    def plt(self, videos=False):
        os.makedirs(self.plt_dir, exist_ok=True)
        csv_file = os.path.join(self.plt_dir, "output", "data", "meta_data.csv")
        img_vector_dir = os.path.join(self.plt_dir, "output", "data", "image-vectors")
        img_dir = os.path.join(self.plt_dir, "output", "data", "images")
        os.makedirs(img_dir)
        os.makedirs(img_vector_dir, exist_ok=True)
        os.system(f"touch {csv_file}")
        with open(csv_file, "a") as fd:
            f_writer = csv.writer(fd)
            f_writer.writerow(["filename", "tags", "description", "Year"])

        if videos:
            vid_dir = os.path.join(self.plt_dir, "vids")
            os.makedirs(vid_dir)
        for index, row in self.data_frame.iterrows():
            if index % 2 == 0:
                scene_name = "".join([str(row[SCENE][0].tolist())]) + "".join(row[NAME])
                self.save_meta(
                    scene_name,
                    row[GENRE],
                    2000,
                    csv_file,
                )

                with open(
                    os.path.join(img_vector_dir, scene_name) + ".jpg" + ".npy", "wb"
                ) as f:
                    data = np.expand_dims(row[T_DATA], 0)
                    print(data.shape)
                    np.save(f, data)
                    print("saved", str(row[SCENE]), row[NAME], row[T_DATA])
                img = row["Image"]
                save_image(img, os.path.join(img_dir, scene_name) + ".jpg")
                if videos:
                    shutil.copy(
                        row[FILEPATH][0], os.path.join(vid_dir, scene_name) + ".mp4"
                    )

        os.system(
            f"pixplot --images '{img_dir}/*.jpg' --metadata '{csv_file}' --use_cache true"
        )
        os.system("python -m http.server 5000")

    def save_meta(self, name, genres, year, csv_fl):
        g = " | ".join(genres)
        filename = name + ".jpg"
        meta_line = [filename, g, name, year]
        with open(csv_fl, "a") as fd:
            f_writer = csv.writer(fd)
            f_writer.writerow(meta_line)
