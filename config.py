VAL_FOLD_ID = 0

IMG_SIZE = 512
SEED = 4213
WARMUP_EPOCH = 1
INIT_LR = 1e-4/3
WARMUP_FACTOR = 10
NUM_WORKERS = 4

BATCH_SIZE = 16 # 64
VAL_BATCH_SIZE = 32
N_EPOCHS = 30

# Debug mode
DEBUG = True  # 10% of total dataset 
DEBUG_SIZE = 0.01

BACKBONE         = "resnet200d"
DATA_PATH        = "/home/argent/kaggle/RANZCR_catheter/dataset/train"
SAVED_MODEL_PATH = "./saved_models/"

TARGET_COLS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

EARLY_STOP = 5
use_amp = False