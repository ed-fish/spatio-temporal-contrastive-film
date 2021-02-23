import pandas as pd
import cv2
import hashlib
import random
import numpy as np
import os
from torchvision import transforms
from preprocessing.customdataloader import GENRE, NAME, FILEPATH, SCENE, O_DATA, T_DATA
import pickle
import matplotlib.pyplot as plt


class Chunk:
    """Convert an array of frames into a transposed
    video "chunk" - Each chunk is a stacked np array of 16 frames.
    Returns: 1 X 16 X 3 (BGR)
    """

    def __init__(self, chunk_array, filepath, mean, std, train_data, norm):
        self.chunk_array = chunk_array
        self.width = chunk_array[0].shape[1]
        self.height = chunk_array[0].shape[0]
        self.filepath = filepath
        self.mean = mean
        self.std = std
        self._init_params()
        self.norm = norm
        self.train_data = train_data

    def _init_params(self):
        """Define random transforms for whole chunk of clips (16 frames)
        Use random seed to ensure all chunks have same transform (temporally consistent)"""
        random.seed(self.gen_hash(self.filepath))
        self.crop_size = random.randrange(100, int(self.height - 20))
        self.x = random.randrange(10, self.width - self.crop_size + 10)
        self.y = random.randrange(10, self.height - self.crop_size + 10)
        self.flip_val = random.randrange(-1, 2)
        self.random_noise = random.randint(1, 10)
        self.random_gray = False
        self.random_blur = False
        # self.noise = np.random.uniform(0, 255,(self.crop_size, self.crop_size))
        if random.random() < 0.50:
            self.random_gray = True
        if random.random() < 0.80:
            self.random_blur = True

    def permutate(self, img):
        # Crop image
        img = img[self.y : self.y + self.crop_size, self.x : self.x + self.crop_size]
        # Flip image
        img = cv2.flip(img, self.flip_val)
        # Generate and add noise
        gaussian_noise = np.zeros_like(img)
        gaussian_noise = cv2.randn(gaussian_noise, 0, self.random_noise)
        img = cv2.add(img, gaussian_noise, dtype=cv2.CV_8UC3)
        if self.random_gray:
            img = cv2.cvtColor(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )
        if self.random_blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        image = self.transform(img)
        return image

    def transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize for resnet - may need variable size depending on backbone
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        # Transpose tensor to (W, H, C)

        tensorfy = transforms.ToTensor()
        img = tensorfy(img)
        # img = img.permute(2, 0, 1)
        if self.norm:
            norm = transforms.Normalize(self.mean, self.std)
            img = norm(img)
        return img

    def chunk_maker(self):
        # todo Convert to tensor

        permuted_chunks = []
        for i, im in enumerate(self.chunk_array):
            if self.train_data:
                permuted_chunks.append(self.permutate(im))
            else:
                permuted_chunks.append(self.transform(im))
            # plt.imshow(trans_im)
            # plt.savefig("/home/ed/PhD/Temporal-3DCNN-pytorch/src/tests/0.png")

        permuted_stack = np.stack(permuted_chunks)
        return permuted_stack

    def gen_hash(self, tbh):
        hash_object = hashlib.md5(tbh.encode())
        return hash_object


class DataTransformer:
    def __init__(self, config, train_data=True, debug=False):
        self.config = config
        self.cache_file = self.config.cache_file
        self.train_data = train_data

    def transform_data_from_cache(self):
        data_frame = pd.read_csv(self.cache_file, delimiter="/")
        column = [0, 1, 2, 3, -1, -2]  # todo Fix to ensure variable length filepaths.
        data_frame.drop(data_frame.columns[column], axis=1, inplace=True)
        data_frame.columns = [GENRE, NAME, SCENE]
        data_frame[FILEPATH] = pd.read_csv(self.cache_file, delimiter="/n")
        print("data frame created")
        self.create_trans_data_frame(
            data_frame, self.config.sample_size, self.config.trans_data_dir
        )

    def split_frames(self, file_path, min_clip_len, n_frames, debug=False):
        count = 0
        frame_list = []
        clip_list = []
        list_of_permuted_imgs = []

        vidcap = cv2.VideoCapture(file_path)
        success, image = vidcap.read()
        frame_list.append(image)

        while success:
            success, image = vidcap.read()
            if success:
                if count % 2:
                    frame_list.append(image)
                if len(frame_list) == n_frames:
                    clip_list.append(frame_list)
                    if debug:
                        print("added chunk of length", len(frame_list))
                        print("added to clip_list", len(clip_list))
                    frame_list = []
                count += 1

        if len(clip_list) >= min_clip_len:
            for i, clip in enumerate(clip_list):
                chunk_obj = Chunk(
                    clip,
                    file_path + str(i),
                    self.config.mean,
                    self.config.std,
                    self.train_data,
                    norm=True,
                )

                p = chunk_obj.chunk_maker()  # returns a list of stacked images
                list_of_permuted_imgs.append(p)

            return list_of_permuted_imgs
        else:
            return 0

    def create_trans_data_frame(self, data_frame, n_samples, save_path):
        print(data_frame)
        if n_samples == 0:
            n_samples = len(data_frame)
        if self.train_data:
            trans_path = os.path.join(save_path, f"{str(n_samples)}_train.pkl")
        else:
            trans_path = os.path.join(save_path, f"{str(n_samples)}_eval.pkl")
        trans_pickle_file_path = open(trans_path, "wb")

        counter = 0
        i = 0
        while counter < n_samples:
            fp = data_frame.at[i, FILEPATH]
            name = data_frame.at[i, NAME]
            scene = data_frame.at[i, SCENE]
            genre = data_frame.at[i, GENRE]
            list_of_transposed_stacks = self.split_frames(fp, 3, 16)
            if list_of_transposed_stacks != 0:
                pickle.dump(
                    [genre, name, fp, scene, list_of_transposed_stacks],
                    trans_pickle_file_path,
                )
                print(f"data added db:{i}, sample{counter}")
                counter += 1

            i += 1

        trans_pickle_file_path.close()


# -> dataframe["filepath", "genre","scene", "original 16 x 3 x
#  112 x 112", ["transform1", "transform2", "transformn"]]
# -> transfrom 1 vs transform 2
# -> transform 3 vs transform 1
# crop size relative to original