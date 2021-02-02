import pandas as pd
import cv2
import os
import hashlib
import random
import numpy as np

class Chunk:
    """Convert an array of frames into a transposed
       video "chunk" - Each chunk is a stacked np array of 16 frames.
       Returns: 1 X 16 X 3 (BGR)
       """
    def __init__(self, chunk_array, filepath):
        self.chunk_array = chunk_array
        self.width =chunk_array[0].shape[1]
        self.height = chunk_array[0].shape[0]
        self.filepath = filepath
        self._init_params()

    def _init_params(self):
        random.seed(self.gen_hash(self.filepath))
        self.crop_size = random.randint(int(int(self.height) / 2), int(self.height))
        self.x = random.randrange(0,self.width - self.crop_size)
        self.y = random.randrange(0, self.height - self.crop_size)
        self.flip_val = random.randrange(-1, 2)
        #self.noise = np.random.uniform(0, 255,(self.crop_size, self.crop_size))
        self.random_gray = random.randrange(1, 6)
        self.random_blur = random.randrange(1, 4)

    def transform(self, image):
        # Crop image
        image = image[self.y:self.y + self.crop_size, self.x:self.x + self.crop_size]
        # Flip image
        image = cv2.flip(image, self.flip_val)
        # Generate and add noise
        #noise = self.generate_noise(image)
        #image = cv2.add(image, noise)
        # Gray scale and blurring
        if self.random_gray == 3: image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if self.random_blur == 3: image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.resize(image, (112, 112), interpolation = cv2.INTER_AREA)
        return image

    def generate_noise(self, image):
        noise = np.random.randint(0,50,(self.crop_size, self.crop_size)) # design jitter/noise here
        zitter = np.zeros_like(image)
        zitter[:,:,1] = noise
        return zitter

    def chunk_maker(self):
        #todo Convert to tensor
        for i, im in enumerate(self.chunk_array):
            trans_im = self.transform(im)
            trans_im = cv2.cvtColor(trans_im, cv2.COLOR_BGR2RGB)
            self.chunk_array[i] = trans_im
        stacked_images = np.stack(self.chunk_array)
        stacked_images = stacked_images.transpose(0,3,1,2)
        return stacked_images

    def gen_hash(self, tbh):
        hash_object = hashlib.md5(tbh.encode())
        return hash_object

def create_data_frame(cache_file, debug=False):
    data_frame = pd.read_csv(cache_file, delimiter='/')
    column = [0,1,2,3, -1, -2]
    data_frame.drop(data_frame.columns[column], axis=1, inplace=True)
    data_frame.columns = ["Genre", "Name", "Scene"]
    data_frame["Filepath"] = pd.read_csv(cache_file, delimiter='/n')
    if debug:
        print('data frame created')
    return data_frame

def split_frames(file_path,out_path, min_clip_len, n_frames, desc, debug=False):
    count = 0
    frame_list = []
    clip_list = []
    stack_list = []

    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    frame_list.append(image)

    while success:
        success,image = vidcap.read()
        if success:
            if count % 2:
                frame_list.append(image)
            if len(frame_list) == n_frames:
                clip_list.append(frame_list)
                if debug:
                    print('added chunk of length', len(frame_list))
                    print('added to clip_list', len(clip_list))
                frame_list = []
            count += 1

    if len(clip_list) >= min_clip_len:
        for i, clip in enumerate(clip_list):
            chunk_pth = os.path.join(out_path, desc)
            name_path = os.path.join(chunk_pth, str(i))
            chunk_obj = Chunk(clip, name_path)
            trans_images = chunk_obj.chunk_maker() #returns a list of stacked images
            stack_list.append(trans_images)
        return stack_list
    else:
        if debug:
            print(len(clip_list))
            print("not enough clips!")
        return 0

def create_trans_data_frame(data_frame):
    data_list = []
    for i in range(30000):
        fp = data_frame.at[i, "Filepath"]
        name = data_frame.at[i, "Name"]
        scene = data_frame.at[i, "Scene"]
        genre = data_frame.at[i, "Genre"]
        try:
            stack_of_chunks = split_frames(fp, "./test/", 3, 16, str(name + str(scene)))
            if type(stack_of_chunks) == int and not stack_of_chunks: continue
            else:
                for i, stack in enumerate(stack_of_chunks):
                    data_list.append([genre, name, scene, fp, stack, i])

        except:
            continue

    transform_df = pd.DataFrame(data_list, columns=['genre', 'name', 'scene', 'filepath', 'data', 'chunk'])
    return transform_df

