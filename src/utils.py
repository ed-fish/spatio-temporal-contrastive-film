import pandas as pd
import pickle as pkl
import numpy as np
import cv2
from PIL import Image


def load_pkl(data_fp, sample_size):
    data_list = []
    pklfl = open(data_fp, "rb")
    i = 0
    while i < sample_size:
        try:
            data_list.append(pkl.load(pklfl))
            print(len(data_list))
            i += 1
        except EOFError:
            break

    pklfl.close()
    data_list = pd.DataFrame(
        data_list, columns=["Genre", "Name", "Scene", "Fp", "Data"]
    )
    return data_list


df = load_pkl("testing.pkl", 10)
clip_one = df["Data"][8][2]
clip_two = df["Data"][0][1]
clip = np.concatenate(clip_one)
vid = []
for i in range(2):
    vc = clip_one[i].transpose(2, 1, 0)
    print(vc)
    img = Image.fromarray(vc, "RGB")
    img.save(f"test{i}.png")
