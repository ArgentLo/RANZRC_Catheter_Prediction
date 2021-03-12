import pandas as pd
import numpy as np
import sys
# sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
# sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import *
from tqdm import tqdm
import timm
from warnings import filterwarnings
filterwarnings("ignore")


from model import RANZCRResNet200D, EffNet_b3, EffNet_b0, RESNEST_50D, REXNET_100
from dataset import RANZERDataset, Transforms_Train, Transforms_Valid
from utils import *
from config import *
from lamb import Lamb, log_lamb_rs


##################################################################
#################     Train and Valid Function     ###############
##################################################################

def train_func(train_loader):

    model.train()
    bar = tqdm(train_loader)
    if USE_AMP:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):

        images, targets = images.to(device), targets.to(device)
        
        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train


def valid_func(valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    PROB = []
    TARGETS = []
    losses = []
    PREDS = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):

            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            PREDS += [logits.sigmoid()]
            TARGETS += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')
            
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    #roc_auc = roc_auc_score(TARGETS.reshape(-1), PREDS.reshape(-1))
    roc_auc = macro_multilabel_auc(TARGETS, PREDS)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc

##################################################################
#################     Train and Valid Function     ###############
##################################################################


# init
device = torch.device('cuda')
seed_everything(SEED)


FOLD_MIN, FOLD_MAX = 0, 4
if RESUME_FOLD:
    FOLD_MIN, FOLD_MAX = RESUME_FOLD, RESUME_FOLD+1

for VAL_FOLD_ID in range(FOLD_MIN, FOLD_MAX):

    # Load train.csv
    df_train = pd.read_csv('./dataset/train_with_folds.csv')
    df_train["file_path"] = df_train.StudyInstanceUID.apply(lambda x: os.path.join(DATA_PATH, f'{x}.png'))  # resize jpg to png

    # DEBUG mode
    if DEBUG:
        df_train = df_train.sample(frac=DEBUG_SIZE)

    # Load image dataset
    dataset = RANZERDataset(df_train, 'train', transform=Transforms_Train)
    df_train_this = df_train[df_train['fold'] != VAL_FOLD_ID]
    df_valid_this = df_train[df_train['fold'] == VAL_FOLD_ID]
    dataset_train = RANZERDataset(df_train_this, 'train', transform=Transforms_Train)
    dataset_valid = RANZERDataset(df_valid_this, 'valid', transform=Transforms_Valid)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Init model
    if BACKBONE == "effnet_b3":
        model = EffNet_b3(out_dim=len(TARGET_COLS), pretrained=True)
    elif BACKBONE == "effnet_b0":
        model = EffNet_b0(out_dim=len(TARGET_COLS), pretrained=True)
    elif BACKBONE == "resnet200d":
        model = RANZCRResNet200D(out_dim=len(TARGET_COLS), pretrained=True)
    elif BACKBONE == "resnest50d":
        model = RESNEST_50D(out_dim=len(TARGET_COLS), pretrained=True)
    elif BACKBONE == "rexnet_100":
        model = REXNET_100(out_dim=len(TARGET_COLS), pretrained=True)

    model = model.to(device)

    # Resume Training
    if RESUME_PATH:
        checkpoint = torch.load(RESUME_PATH)
        model.load_state_dict(checkpoint)
        print('>>> Resume Training. Saved Model loaded {}'.format(RESUME_PATH), "\n")

    # Optimization
    ClassWeights = ClassWeights.to(device)
    if use_FocalLoss:
        print(">>> Training with FocalLoss")
        criterion = FocalLoss(alpha=2, gamma=2, pos_weight=ClassWeights)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=ClassWeights)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    # optimizer = Lamb(model.parameters(), lr=INIT_LR, weight_decay=0, betas=(.9, .999))
    if DataParallel:
        model = nn.DataParallel(model)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, COSINE_EPO, eta_min=1e-7)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int((COSINE_EPO)/7), T_mult=2, eta_min=2e-7)
    if not RESUME_PATH:
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=WARMUP_Multiplier, total_epoch=WARMUP_EPOCH, after_scheduler=scheduler_cosine)

    # For logging
    log = {}
    roc_auc_max = 0.
    loss_min = 99999
    not_improving = 0

    ##########################################################################

    # Training Loop
    for epoch in range(1, N_EPOCHS+1):
        if not RESUME_PATH:
            scheduler_warmup.step(epoch-1)
        else: 
            scheduler_cosine.step(epoch-1)
        loss_train = train_func(train_loader)
        loss_valid, roc_auc = valid_func(valid_loader)

        log['loss_train'] = log.get('loss_train', []) + [loss_train]
        log['loss_valid'] = log.get('loss_valid', []) + [loss_valid]
        log['lr'] = log.get('lr', []) + [optimizer.param_groups[0]["lr"]]
        log['roc_auc'] = log.get('roc_auc', []) + [roc_auc]

        content = ">>>>>>  " + time.ctime() + '\n' + f'Fold {VAL_FOLD_ID}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, roc_auc: {roc_auc:.6f}.'
        print(content)
        not_improving += 1
        
        if roc_auc > roc_auc_max:
            print(f'roc_auc_max ({roc_auc_max:.6f} --> {roc_auc:.6f}). Saving model ...')
            if use_FocalLoss:
                save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_Focal_best_AUC.pth'
            else: 
                save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_best_AUC.pth'
            torch.save(model.state_dict(), save_name)
            roc_auc_max = roc_auc
            not_improving = 0

        if loss_valid < loss_min:
            loss_min = loss_valid
            if use_FocalLoss:
                save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_Focal_best_loss.pth'
            else: 
                save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_best_loss.pth'
            torch.save(model.state_dict(), save_name)

        print("###"*20 + "\n")
            
        if not_improving == EARLY_STOP:
            print('Early Stopping...')
            break

        # run 1 epoch for Debug mode
        if DEBUG:
            pass
            # break

    # Save final model
    if use_FocalLoss:
        save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_Focal_final.pth'
    else: 
        save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_final.pth'
    torch.save(model.state_dict(), save_name)
