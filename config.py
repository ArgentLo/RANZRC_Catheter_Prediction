import os

#### Regular args
BATCH_SIZE   = 28  # 64
WARMUP_EPOCH = 1
COSINE_EPO   = 35
N_EPOCHS     = WARMUP_EPOCH + COSINE_EPO  # 30
INIT_LR      = 4e-4  # 5e-4
EARLY_STOP   = 7  # if ReStart, should give longer epochs to find minimun after Great LR.  
RESUME_FOLD  = None  # None
RESUME_PATH  = None  #"./saved_models/resnet200d_Fold{}_best_AUC.pth".format(RESUME_FOLD)

#### Backbone
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# BACKBONE = "effnet_b3"
# BACKBONE = "effnet_b0"
# BACKBONE = "hrnet_w40"  # HRNet
BACKBONE = "hrnet_w64"
# BACKBONE = "resnet200d"


#### Debug mode
DEBUG = False  # 10% of total dataset 
DEBUG_SIZE = 0.10


###############################################
###############################################
DATA_PATH        = "/home/argent/kaggle/RANZCR_catheter/dataset/train"
SAVED_MODEL_PATH = "./saved_models/"

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']


USE_AMP = True
DataParallel = False
IMG_SIZE = 512
SEED = 4213
VAL_BATCH_SIZE = 32
NUM_WORKERS = 8
WARMUP_Multiplier = 10
