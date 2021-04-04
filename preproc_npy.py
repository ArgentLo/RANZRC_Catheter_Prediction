import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd


csv_file = pd.read_csv("./dataset/train.csv")

RESIZE_DIM = (1024, 1024)
npy_list = []
img_paths = csv_file["StudyInstanceUID"].values
for img in tqdm(img_paths):
    
    img_path = f"./dataset/train/{img}.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # resize image
    resized = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)
    npy_list.append(resized)

npy_list = np.array(npy_list)
print(">>> Output size: ", npy_list.shape)

# save as npy
OUT_PATH_1024 = "./dataset/train_1024x1024.npy" 
np.save(OUT_PATH_1024, npy_list)