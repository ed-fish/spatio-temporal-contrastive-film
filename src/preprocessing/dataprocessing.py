import pandas as pd
import cv2
import os
import hashlib
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle


class Chunk:
    """Convert an array of frames into a transposed
    video "chunk" - Each chunk is a stacked np array of 16 frames.
    Returns: 1 X 16 X 3 (BGR)
    """

    def __init__(self, chunk_array, filepath):
        self.chunk_array = chunk_array
        self.width = chunk_array[0].shape[1]
        self.height = chunk_array[0].shape[0]
        self.filepath = filepath
        self._init_params()

    def _init_params(self):
        random.seed(self.gen_hash(self.filepath))
        self.crop_size = random.randint(int(int(self.height) / 2), int(self.height))
        self.x = random.randrange(0, self.width - self.crop_size)
        self.y = random.randrange(0, self.height - self.crop_size)
        self.flip_val = random.randrange(-1, 2)
        # self.noise = np.random.uniform(0, 255,(self.crop_size, self.crop_size))
        self.random_gray = random.randrange(1, 6)
        self.random_blur = random.randrange(1, 4)

    def transform(self, img):
        # Crop image
        image = img[self.y : self.y + self.crop_size, self.x : self.x + self.crop_size]
        # Flip image
        image = cv2.flip(img, self.flip_val)
        # Generate and add noise
        # noise = self.generate_noise(img)
        # image = cv2.add(img, noise)
        # Gray scale and blurring
        if self.random_gray == 3:
            img = cv2.cvtColor(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )
        if self.random_blur == 3:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        tensorfy = transforms.ToTensor()
        norm = transforms.Normalize(
            (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
        )
        img = norm(tensorfy(img))

        return img

    def generate_noise(self, image):
        noise = np.random.randint(
            0, 50, (self.crop_size, self.crop_size)
        )  # design jitter/noise here
        zitter = np.zeros_like(image)
        zitter[:, :, 1] = noise
        return zitter

    def chunk_maker(self):
        # todo Convert to tensor
        for i, im in enumerate(self.chunk_array):
            trans_im = self.transform(im)
            self.chunk_array[i] = trans_im
        stacked_images = np.stack(self.chunk_array)
        return stacked_images

    def gen_hash(self, tbh):
        hash_object = hashlib.md5(tbh.encode())
        return hash_object


def create_data_frame(cache_file, debug=False):
    data_frame = pd.read_csv(cache_file, delimiter="/")
    column = [0, 1, 2, 3, -1, -2]
    data_frame.drop(data_frame.columns[column], axis=1, inplace=True)
    data_frame.columns = ["Genre", "Name", "Scene"]
    data_frame["Filepath"] = pd.read_csv(cache_file, delimiter="/n")
    if debug:
        print("data frame created")
    return data_frame


def split_frames(file_path, min_clip_len, n_frames, debug=False):
    count = 0
    frame_list = []
    clip_list = []
    stack_list = []

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
            print(file_path)
            chunk_obj = Chunk(clip, file_path + str(i))
            trans_images = chunk_obj.chunk_maker()  # returns a list of stacked images
            stack_list.append(trans_images)
        return stack_list
    else:
        return 0


def create_trans_data_frame(data_frame, n_samples, save_path):
    if n_samples == 0:
        n_samples = len(data_frame)
    f = open(save_path, "wb")
    for i in range(0, n_samples):
        fp = data_frame.at[i, "Filepath"]
        name = data_frame.at[i, "Name"]
        scene = data_frame.at[i, "Scene"]
        genre = data_frame.at[i, "Genre"]
        try:
            stack_of_chunks = split_frames(fp, 3, 16)
            if stack_of_chunks == 0:
                # print('fail')
                continue
            else:
                pickle.dump([genre, name, scene, fp, stack_of_chunks], f)
                print("data added", i)

        except ValueError:
            print("Value error")
            continue
    f.close()
