import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import *
from warnings import filterwarnings
filterwarnings("ignore")


from config import *


class RANZERDataset(Dataset):

    def __init__(self, df, mode, transform=None):
        
        # print("DATA PATH:", df.iloc[0])
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.labels = df[TARGET_COLS].values  # 11 cols to predict
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.loc[index]

        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
                
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        label = torch.tensor(self.labels[index]).float()
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), label



######################################################
#################     Augmentation     ###############
######################################################


# # Plain Training Augmentation
# Transforms_Train = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
#     A.Normalize()
# ])



# Training Augmentation
Transforms_Train = A.Compose([

    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.9, 1), p=1), 
    A.HorizontalFlip(p=0.5),

    # Brightness + Contract
    A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.75),

    # Blurring + Distortion
    A.OneOf([
        A.GaussNoise(var_limit=[5.0, 30.0]), A.MotionBlur(blur_limit=5), 
        A.MedianBlur(blur_limit=5), A.GaussianBlur(blur_limit=5)], p=0.75),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0), A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3)], p=0.75),

    # Some final Shift+Saturation
    A.CLAHE(clip_limit=(1,4), p=0.75),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.85),

    # Resize
    A.Resize(IMG_SIZE, IMG_SIZE),

    # # New from HeQishen's kernel
    # A.OneOf([
    #     JpegCompression(),
    #     Downscale(scale_min=0.1, scale_max=0.15),
    # ], p=0.2),
    # IAAPiecewiseAffine(p=0.2),
    # IAASharpen(p=0.2),

    # cut holes on imgs
    A.Cutout(max_h_size=int(IMG_SIZE * 0.10), max_w_size=int(IMG_SIZE * 0.10), num_holes=5, p=0.75),
    A.Normalize(),
])


# Validation Augmentation
Transforms_Valid = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize()
])

######################################################
#################     Augmentation     ###############
######################################################