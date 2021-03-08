IMG_SIZE = 512
SEED = 4213
WARMUP_EPOCH = 1
INIT_LR = 4e-4
WARMUP_FACTOR = 10
NUM_WORKERS = 4

BATCH_SIZE = 18 # 64
VAL_BATCH_SIZE = 32
N_EPOCHS = 30


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BACKBONE = "effnet_b5"
# BACKBONE = "resnet200d"


# Debug mode
DEBUG = False  # 10% of total dataset 
DEBUG_SIZE = 0.05

###############################################


DATA_PATH        = "/home/argent/kaggle/RANZCR_catheter/dataset/train"
SAVED_MODEL_PATH = "./saved_models/"

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

EARLY_STOP = 5
USE_AMP = True
DataParallel = False