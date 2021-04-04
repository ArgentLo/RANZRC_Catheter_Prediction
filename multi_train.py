from config import DEBUG
import gc
import copy
import yaml
import numpy as np
import pandas as pd

import torch
from torch import nn
import pytorch_pfn_extras as ppe

print("Import")
from multi_config import *
from multi_utils  import *

##################################################
# Numpy dataset
##################################################

train_data_arr = np.load("./dataset/train_512x512.npy", mmap_mode="r")
train = pd.read_csv(DATA / "train.csv")
smpl_sub =  pd.read_csv(DATA / "sample_submission.csv")

##################################################
# Split data
##################################################

label_arr = train[CLASSES].values
group_id = train.PatientID.values

train_val_indexs = list(multi_label_stratified_group_k_fold(label_arr, group_id, N_FOLD, RANDAM_SEED))

train["fold"] = -1
for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
    train.loc[val_idx, "fold"] = fold_id

##################################################
# Train Function
##################################################

def run_train_loop(manager, stgs, model, device, train_loader, optimizer, scheduler, loss_func, VAL_FOLD_ID):
    """Run minibatch training loop"""
    step_scheduler_by_epoch, step_scheduler_by_iter = get_stepper(manager, stgs, scheduler)

    if stgs["globals"]["use_amp"]:
        print(">>> Using AMP Training...")

        EPOCH = 1
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

            # save every epoch
            save_name = f'{SAVED_MODEL_PATH}{BACKBONE}_Fold{VAL_FOLD_ID}_Size{IMG_SIZE}_Epoch{EPOCH}.pth'
            torch.save(model.state_dict(), save_name)
            EPOCH += 1

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

def train_one_fold(settings, train_all, output_path, VAL_FOLD_ID, print_progress=False):
    """train one fold"""
    torch.backends.cudnn.benchmark = True
    set_random_seed(settings["globals"]["seed"])

    # prepare train, valid paths
    train_file_list, val_file_list = get_file_list_with_array(settings, train_all)
    print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))
    device = torch.device(settings["globals"]["device"])
    
    # get data_loader
    train_loader, val_loader = get_dataloaders_cls(
        settings, train_file_list, val_file_list, LabeledImageDatasetNumpy)

    # get model
    # model = MultiHead_DenseNet121(**settings["model"]["params"])
    # model = MultiHead_SeResNet152d(**settings["model"]["params"])
    # model = MultiHeadResNet200D(**settings["model"]["params"])
    # model = MultiHead_EffNetb5(**settings["model"]["params"])
    model = MultiHead_InceptionV3(**settings["model"]["params"])
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

    # get loss
    ClassWeights = torch.tensor([1 for _ in range(7)] + [2]*3 + [1])
    ClassWeights = ClassWeights.to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=ClassWeights)
    loss_func.to(device)

    eval_manager = EvalFuncManager(
        len(val_loader), {
            metric["report_name"]: eval(metric["name"])(**metric["params"])
            for metric in settings["eval"]
        })
    eval_manager.to(device)

    # get manager
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

    # run training.
    run_train_loop(manager, settings, model, device, train_loader,
                   optimizer, scheduler, loss_func, VAL_FOLD_ID)



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

    print(f">>> Current Fold_ID: {fold_id}")

    train_one_fold(tmp_stgs, train, TMP / f"fold{fold_id}", VAL_FOLD_ID=fold_id, print_progress=True)
    torch.cuda.empty_cache()
    gc.collect()