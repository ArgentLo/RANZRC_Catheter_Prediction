import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
IMG_SIZE = 512
BACKBONE =  "MultiHead_DenseNet121"  # "MultiHead_SeResNet152d" "MultiHeadResNet200D" "MultiHead_EffNetb5"
DEBUG = False

ROOT   = Path.cwd()
# INPUT  = ROOT / "input"
# OUTPUT = ROOT / "output"
DATA   = ROOT / "dataset"
TRAIN  = DATA / "train"
TEST   = DATA / "test"

TRAIN_NPY = DATA
TMP = ROOT / "saved_models"

stgs_str = """
globals:
  seed: 4213
  device: cuda
  max_epoch: 15
  patience: 3
  dataset_name: train_512x512
  use_amp: True
  val_fold: 0
  debug: False

dataset:
  name: LabeledImageDatasetNumpy
  train:
    transform_list:
      - [HorizontalFlip, {p: 0.5}]
      - [ShiftScaleRotate, {
          p: 0.65, shift_limit: 0.2, scale_limit: 0.2,
          rotate_limit: 20, border_mode: 0, value: 0, mask_value: 0}]
      - [RandomResizedCrop, {height: 512, width: 512, scale: [0.8, 1.2]}]
      - [Cutout, {max_h_size: 51, max_w_size: 51, num_holes: 5, p: 0.5}]
      - [Normalize, {
          always_apply: True, max_pixel_value: 255.0,
          mean: [0.4887381077884414], std: [0.23064819430546407]}]
      - [ToTensorV2, {always_apply: True}]
  val:
    transform_list:
      - [Normalize, {
          always_apply: True, max_pixel_value: 255.0,
          mean: [0.4887381077884414], std: [0.23064819430546407]}]
      - [ToTensorV2, {always_apply: True}]

loader:
  train: {batch_size: 64, shuffle: True, num_workers: 8, pin_memory: True, drop_last: True}
  val: {batch_size: 16, shuffle: False, num_workers: 2, pin_memory: True, drop_last: False}

model:
  name: MultiHead_InceptionV3 # MultiHead_DenseNet121 MultiHeadResNet200D MultiHead_SeResNet152d MultiHead_EffNetb5
  params:
    out_dims_head: [3, 4, 3, 1]
    pretrained: True

loss: {name: BCEWithLogitsLoss, params: {}}

eval:
  - {name: MyLogLoss, report_name: loss, params: {}}
  - {name: MyROCAUC, report_name: metric, params: {average: macro}}

optimizer:
    name: Adam
    params:
      lr: 6.5e-05  # ResNet200d: 2.5e-04 || SeResNet: 1.e-04

scheduler:
  name: CosineAnnealingWarmRestarts
  params:
    T_0: 16
    T_mult: 1
"""


RANDAM_SEED = 4213
N_CLASSES = 11
# FOLDS = [0, 1, 2, 3, 4]
# N_FOLD = len(FOLDS)
FOLDS = [0, 2, 4]
N_FOLD = 5
SAVED_MODEL_PATH = "./saved_models/"
CLASSES = [
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
    'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
]