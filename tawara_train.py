import gc
import os
import sys
import time
import copy
import random
import shutil
import typing as tp
from pathlib import Path
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from joblib import Parallel, delayed

import cv2
import albumentations

from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils import data
from torchvision import models as torchvision_models
import timm
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions

from tawara_config import *
from tawara_utils  import *

##################################################
# Numpy dataset
##################################################

train_data_arr = np.load("./dataset/train_640x640.npy", mmap_mode="r")

for p in DATA.iterdir():
    print(p.name)

train = pd.read_csv(DATA / "train.csv")
smpl_sub =  pd.read_csv(DATA / "sample_submission.csv")

##################################################
# Split data
##################################################

label_arr = train[CLASSES].values
group_id = train.PatientID.values

train_val_indexs = list(
    multi_label_stratified_group_k_fold(label_arr, group_id, N_FOLD, RANDAM_SEED))

train["fold"] = -1
for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
    train.loc[val_idx, "fold"] = fold_id
    
train.groupby("fold")[CLASSES].sum()


##################################################
# forward test
##################################################

# m = SingleHeadModel("resnext50_32x4d", 11, True)
m = MultiHeadResNet200D([3, 4, 3, 1], True)
m = m.eval()

x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    y = m(x)
print("[forward test]")
print("input:\t{}\noutput:\t{}".format(x.shape, y.shape))

del m; del x; del y
gc.collect()

##################################################
# Train Function
##################################################

def run_train_loop(
    manager, stgs, model, device, train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    step_scheduler_by_epoch, step_scheduler_by_iter = get_stepper(manager, stgs, scheduler)

    if stgs["globals"]["use_amp"]:     
        while not manager.stop_trigger:
            model.train()
            scaler = torch.cuda.amp.GradScaler()
            for x, t in train_loader:
                with manager.run_iteration():
                    x, t = x.to(device), t.to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        y = model(x)
                        loss = loss_func(y, t)
                    ppe.reporting.report({'train/loss': loss.item()})
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    step_scheduler_by_iter()
            step_scheduler_by_epoch()
    else:
        while not manager.stop_trigger:
            model.train()
            for x, t in train_loader:
                with manager.run_iteration():
                    x, t = x.to(device), t.to(device)
                    optimizer.zero_grad()
                    y = model(x)
                    loss = loss_func(y, t)
                    ppe.reporting.report({'train/loss': loss.item()})
                    loss.backward()
                    optimizer.step()
                    step_scheduler_by_iter()
            step_scheduler_by_epoch()
        
        
def run_eval(stgs, model, device, batch, eval_manager):
    """Run evaliation for val or test. this function is applied to each batch."""
    model.eval()
    x, t = batch
    if stgs["globals"]["use_amp"]:
        with torch.cuda.amp.autocast(): 
            y = model(x.to(device))
            eval_manager(y, t.to(device))
    else:
        y = model(x.to(device))
        eval_manager(y, t.to(device))


##################################################
# Train One Fold
##################################################

def train_one_fold(settings, train_all, output_path, print_progress=False):
    """train one fold"""
    torch.backends.cudnn.benchmark = True
    set_random_seed(settings["globals"]["seed"])

    # # prepare train, valid paths
    # train_file_list, val_file_list = get_file_list(settings, train_all, "png")
    train_file_list, val_file_list = get_file_list_with_array(settings, train_all)
    print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))

    device = torch.device(settings["globals"]["device"])
    # # get data_loader
    train_loader, val_loader = get_dataloaders_cls(
        settings, train_file_list, val_file_list, LabeledImageDatasetNumpy)

    # # get model
    model = MultiHeadResNet200D(**settings["model"]["params"])
    model.to(device)

    # # get optimizer
    optimizer = getattr(
        torch.optim, settings["optimizer"]["name"]
    )(model.parameters(), **settings["optimizer"]["params"])

    # # get scheduler
    if settings["scheduler"]["name"] == "OneCycleLR":
        settings["scheduler"]["params"]["epochs"] = settings["globals"]["max_epoch"]
        settings["scheduler"]["params"]["steps_per_epoch"] = len(train_loader)
    scheduler = getattr(
        torch.optim.lr_scheduler, settings["scheduler"]["name"]
    )(optimizer, **settings["scheduler"]["params"])

    # # get loss
    if hasattr(nn, settings["loss"]["name"]):
        loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    else:
        loss_func = eval(settings["loss"]["name"])(**settings["loss"]["params"])
    loss_func.to(device)

    eval_manager = EvalFuncManager(
        len(val_loader), {
            metric["report_name"]: eval(metric["name"])(**metric["params"])
            for metric in settings["eval"]
        })
    eval_manager.to(device)

    # # get manager
    # trigger = None
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        check_trigger=(1, 'epoch'),
        # monitor='val/metric', mode="min",
        monitor='val/metric', mode="max",
        patience=settings["globals"]["patience"], verbose=False,
        max_trigger=(settings["globals"]["max_epoch"], 'epoch'),
    )
    manager = ppe.training.ExtensionsManager(
        model, optimizer, settings["globals"]["max_epoch"],
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger, out_dir=output_path
    )
    manager = set_extensions(
        manager, settings, model, device, val_loader, optimizer, eval_manager, print_progress)

    # # run training.
    run_train_loop(
        manager, settings, model, device, train_loader,
        optimizer, scheduler, loss_func)


##################################################
# Read config settings
##################################################

stgs = yaml.safe_load(stgs_str)

if stgs["globals"]["debug"]:
    stgs["globals"]["max_epoch"] = 1


##################################################
# Read config settings
##################################################

stgs_list = []
for fold_id in FOLDS:
    tmp_stgs = copy.deepcopy(stgs)
    tmp_stgs["globals"]["val_fold"] = fold_id
    stgs_list.append(tmp_stgs)


##################################################
# Train Loops
##################################################

torch.cuda.empty_cache()
gc.collect()

for fold_id, tmp_stgs in zip(FOLDS, stgs_list):
    train_one_fold(tmp_stgs, train, TMP / f"fold{fold_id}", False)
    torch.cuda.empty_cache()
    gc.collect()


##################################################
# Eval OOF and Get the BEST Model
##################################################

best_log_list = []
for fold_id, tmp_stgs in zip(FOLDS, stgs_list):
    exp_dir_path = TMP / f"fold{fold_id}"
    log = pd.read_json(exp_dir_path / "log")
    best_log = log.iloc[[log["val/metric"].idxmax()],]
    best_epoch = best_log.epoch.values[0]
    best_log_list.append(best_log)
    
    best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
    copy_to = f"./best_model_fold{fold_id}.pth"
    shutil.copy(best_model_path, copy_to)
    
    for p in exp_dir_path.glob("*.pth"):
        p.unlink()
    
    shutil.copytree(exp_dir_path, f"./fold{fold_id}")
    
    with open(f"./fold{fold_id}/settings.yml", "w") as fw:
        yaml.dump(tmp_stgs, fw)
    
pd.concat(best_log_list, axis=0, ignore_index=True)


##################################################
# 
##################################################