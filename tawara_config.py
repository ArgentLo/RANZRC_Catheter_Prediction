import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path


ROOT   = Path.cwd()
# INPUT  = ROOT / "input"
# OUTPUT = ROOT / "output"
DATA   = ROOT / "dataset"
TRAIN  = DATA / "train"
TEST   = DATA / "test"

TRAIN_NPY = DATA
TMP = ROOT / "saved_models"

RANDAM_SEED = 4213
N_CLASSES = 11
# FOLDS = [0, 1, 2, 3, 4]
# N_FOLD = len(FOLDS)
FOLDS = [0, 2, 4]
N_FOLD = 5

CLASSES = [
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
    'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
]

stgs_str = """
globals:
  seed: 1086
  device: cuda
  max_epoch: 16
  patience: 3
  dataset_name: train_640x640
  use_amp: True
  val_fold: 0
  debug: False

dataset:
  name: LabeledImageDatasetNumpy
  train:
    transform_list:
      - [HorizontalFlip, {p: 0.5}]
      - [ShiftScaleRotate, {
          p: 0.5, shift_limit: 0.2, scale_limit: 0.2,
          rotate_limit: 20, border_mode: 0, value: 0, mask_value: 0}]
      - [RandomResizedCrop, {height: 512, width: 512, scale: [0.9, 1.0]}]
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
  train: {batch_size: 8, shuffle: True, num_workers: 2, pin_memory: True, drop_last: True}
  val: {batch_size: 32, shuffle: False, num_workers: 2, pin_memory: True, drop_last: False}

model:
  name: MultiHeadResNet200D
  params:
    # base_name: resnet200D_320
    out_dims_head: [3, 4, 3, 1]
    pretrained: True

loss: {name: BCEWithLogitsLoss, params: {}}

eval:
  - {name: MyLogLoss, report_name: loss, params: {}}
  - {name: MyROCAUC, report_name: metric, params: {average: macro}}

optimizer:
    name: Adam
    params:
      lr: 2.5e-04

scheduler:
  name: CosineAnnealingWarmRestarts
  params:
    T_0: 16
    T_mult: 1
"""






