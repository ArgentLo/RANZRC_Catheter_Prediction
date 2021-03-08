import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import albumentations
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


# Training Augmentation
Transforms_Train = albumentations.Compose([
    albumentations.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.9, 1), p=1), 
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
    albumentations.CLAHE(clip_limit=(1,4), p=0.5),
    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.0),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        albumentations.ElasticTransform(alpha=3),
    ], p=0.2),
    albumentations.OneOf([
        albumentations.GaussNoise(var_limit=[10, 50]),
        albumentations.GaussianBlur(),
        albumentations.MotionBlur(),
        albumentations.MedianBlur(),
    ], p=0.2),
    albumentations.Resize(IMG_SIZE, IMG_SIZE),
    albumentations.OneOf([
        JpegCompression(),
        Downscale(scale_min=0.1, scale_max=0.15),
    ], p=0.2),
    IAAPiecewiseAffine(p=0.2),
    IAASharpen(p=0.2),
    albumentations.Cutout(max_h_size=int(IMG_SIZE * 0.1), max_w_size=int(IMG_SIZE * 0.1), num_holes=5, p=0.5),
    albumentations.Normalize(),
])


# Validation Augmentation
Transforms_Valid = albumentations.Compose([
    albumentations.Resize(IMG_SIZE, IMG_SIZE),
    albumentations.Normalize()
])

######################################################
#################     Augmentation     ###############
######################################################