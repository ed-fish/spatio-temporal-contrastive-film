
"""Data has bad format - Requires reference csv for retrieval
   # todo Construct csv for all videos - fields must include - name, genre, scene, clip_filepath, image
   # todo Process all videos - extracting frames - do transforms - remove excess data - store references
   # todo Gather pretrained model - resnet 3d CNN
   # todo train unsupervised end to end with transforms
   # todo TOP SECRET STUFF I CANT TELL YOU ABOUT

"""
from src.preprocessing import dataprocessing as dp
import pandas as pd

def create_data_frame(text_file, out_pkl):
    data_frame = dp.create_data_frame(text_file, debug=True)
    transform_df = dp.create_trans_data_frame(data_frame)
    transform_df.to_pickle(out_pkl)

def read_data_frame(pkl_file):
    return pd.read_pickle(pkl_file)

def main():
    print("processing data")

if __name__ == "__main__":
    main()