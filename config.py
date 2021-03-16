import os
import torch

#### Regular args
BATCH_SIZE   = 12  # 64
WARMUP_EPOCH = 1
COSINE_EPO   = 20
N_EPOCHS     = WARMUP_EPOCH + COSINE_EPO  # 30
INIT_LR      = 3e-5  # 5e-4
EARLY_STOP   = 5  # if ReStart, should give longer epochs to find minimun after Great LR.
use_FocalLoss = False
ClassWeights  = torch.tensor([1 for _ in range(7)] + [2]*3 + [1])
# ClassWeights  = torch.tensor([1 for _ in range(11)])

# Image size
IMG_SIZE = 1024  # 512, 640, 1024
DATA_PATH = "/home/argent/kaggle/RANZCR_catheter/dataset/train_resized_{}".format(IMG_SIZE)

# Resume
RESUME_FOLD  = 0  # None
RESUME_PATH  = None  #"./saved_models/resnet200d_Fold{}_best_AUC.pth".format(RESUME_FOLD)

#### Backbone
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BACKBONE = "effnet_b3"
# BACKBONE = "effnet_b0"
# BACKBONE = "resnest50d"  # resnest
# BACKBONE = "rexnet_100"  # rexnet
# BACKBONE = "resnet200d"


#### Debug mode
DEBUG = False  # 10% of total dataset 
DEBUG_SIZE = 0.10


###############################################
###############################################
SAVED_MODEL_PATH = "./saved_models/"

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']


USE_AMP = True
DataParallel = False
SEED = 9527
VAL_BATCH_SIZE = 16
NUM_WORKERS = 8
WARMUP_Multiplier = 10
