import gc
import os
import time
import random
import typing as tp
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

import torch
from torch import nn
from torch.utils import data
import timm
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

from tawara_config import *

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
##############    Numpy Dataloader    ############
##################################################

def get_file_list_with_array(stgs, train_all):
    """Get file path and target info."""
    # train_all = pd.read_csv(DATA / stgs["globals"]["meta_file"])
    use_fold = stgs["globals"]["val_fold"]
    
    train_idx = train_all[train_all["fold"] != use_fold].index.values
    if stgs["globals"]["debug"]:
        train_idx = train_idx[:len(train_idx) // 20]
    val_idx = train_all[train_all["fold"] == use_fold].index.values
    
    train_data_path = TRAIN_NPY / "{}.npy".format(stgs["globals"]["dataset_name"])
    print(train_data_path)
    # train_data_arr = np.load(train_data_path)
    train_data_arr = np.load(train_data_path, mmap_mode="r")
    label_arr = train_all[CLASSES].values.astype("f")
    print(train_data_arr.shape, label_arr.shape)

    train_file_list = [
        (train_data_arr[idx][..., None], label_arr[idx])  for idx in train_idx]


    val_file_list = [
        (train_data_arr[idx][..., None], label_arr[idx])  for idx in val_idx]

    if DEBUG:
        train_file_list = train_file_list[:50]
        val_file_list   = val_file_list[:50]
    print(">>>>>>>>  Train Examples: ", len(train_file_list))

    return train_file_list, val_file_list


def get_dataloaders_cls(
    stgs: tp.Dict,
    train_file_list: tp.List[tp.List],
    val_file_list: tp.List[tp.List],
    dataset_class: data.Dataset
):
    """Create DataLoader"""
    train_loader = val_loader = None
    if train_file_list is not None:
        train_dataset = dataset_class(
            train_file_list, **stgs["dataset"]["train"])
        train_loader = data.DataLoader(
            train_dataset, **stgs["loader"]["train"])

    if val_file_list is not None:
        val_dataset = dataset_class(
            val_file_list, **stgs["dataset"]["val"])
        val_loader = data.DataLoader(
            val_dataset, **stgs["loader"]["val"])

    return train_loader, val_loader



class LabeledImageDatasetNumpy(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[
            tp.Tuple[np.ndarray, tp.Union[int, float, np.ndarray]]],
        transform_list: tp.List[tp.Dict],
        copy_in_channels=True, in_channels=3,
    ):
        """Initialize"""
        self.file_list = file_list
        self.transform = ImageTransformForCls(transform_list)
        self.copy_in_channels = copy_in_channels
        self.in_channels = in_channels 

    def __len__(self):
        """Return Num of Images."""
        return len(self.file_list)

    def __getitem__(self, index):
        """Return transformed image and mask for given index."""
        img, label = self.file_list[index]
        if img.shape[-1] == 2:
            img = img[..., None]

        if self.copy_in_channels:
            img = np.repeat(img, self.in_channels, axis=2)
        
        img, label = self.transform((img, label))
        return img, label



##################################################
################    Custom Model    ##############
##################################################

class SingleHeadModel(nn.Module):
    
    def __init__(
        self, base_name: str='resnext50_32x4d', out_dim: int=11, pretrained=False
    ):
        """"""
        self.base_name = base_name
        super(SingleHeadModel, self).__init__()
        
        # # load base model
        base_model = timm.create_model(base_name, pretrained=pretrained)
        in_features = base_model.num_features
        
        # # remove global pooling and head classifier
        # base_model.reset_classifier(0, '')
        base_model.reset_classifier(0)
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Single Heads.
        self.head_fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, out_dim))

    def forward(self, x):
        """"""
        h = self.backbone(x)
        h = self.head_fc(h)
        return h
        

class MultiHeadModel(nn.Module):
    
    def __init__(
        self, base_name: str='resnext50_32x4d',
        out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = base_name
        self.n_heads = len(out_dims_head)
        super(MultiHeadModel, self).__init__()
        
        # # load base model
        base_model = timm.create_model(base_name, pretrained=pretrained)
        in_features = base_model.num_features
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


#######################################################################################
#######################################################################################
#######################################################################################

    
class MultiHead_DenseNet121(nn.Module):
    
    def __init__(self, out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = "densenet121"
        self.n_heads = len(out_dims_head)
        super(MultiHead_DenseNet121, self).__init__()
        
        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)
        in_features = base_model.num_features
        
        if pretrained:
            print("\n>>>>>>>>>>>\nLoading Pretrained Models\n>>>>>>>>>>>\n")
            pretrained_model_path = './saved_models/densenet121_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


#######################################################################################
#######################################################################################
#######################################################################################

    
class MultiHead_InceptionV3(nn.Module):
    
    def __init__(self, out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = "inception_v3"
        self.n_heads = len(out_dims_head)
        super(MultiHead_InceptionV3, self).__init__()
        
        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)
        in_features = base_model.num_features
        
        if pretrained:
            print("\n>>>>>>>>>>>\nLoading Pretrained Models\n>>>>>>>>>>>\n")
            pretrained_model_path = './saved_models/inception_v3_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


#######################################################################################
#######################################################################################
#######################################################################################

    
class MultiHead_EffNetb5(nn.Module):
    
    def __init__(self, out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = "efficientnet_b5"
        self.n_heads = len(out_dims_head)
        super(MultiHead_EffNetb5, self).__init__()
        
        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)
        in_features = base_model.num_features
        
        if pretrained:
            print("\n>>>>>>>>>>>\nLoading Pretrained Models\n>>>>>>>>>>>\n")
            pretrained_model_path = './saved_models/tf_efficientnet_b5_ns_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


#######################################################################################
#######################################################################################
#######################################################################################

class MultiHead_SeResNet152d(nn.Module):
    
    def __init__(self, out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = "seresnet152d"
        self.n_heads = len(out_dims_head)
        super(MultiHead_SeResNet152d, self).__init__()
        
        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)
        in_features = base_model.num_features
        
        if pretrained:
            print("\n>>>>>>>>>>>\nLoading Pretrained Models\n>>>>>>>>>>>\n")
            pretrained_model_path = './saved_models/seresnet152d_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


    
class MultiHeadResNet200D(nn.Module):
    
    def __init__(
        self, out_dims_head: tp.List[int]=[3, 4, 3, 1], pretrained=False
    ):
        """"""
        self.base_name = "resnet200d_320"
        self.n_heads = len(out_dims_head)
        super(MultiHeadResNet200D, self).__init__()
        
        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)
        in_features = base_model.num_features
        
        if pretrained:
            print("\n>>>>>>>>>>>\nLoading Pretrained Models\n>>>>>>>>>>>\n")
            pretrained_model_path = './saved_models/resnet200d_320_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)
        
        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


##################################################
################    Custom Block    ##############
##################################################

def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError
        

class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)
        

class SSEBlock(nn.Module):
    """channel `S`queeze and `s`patial `E`xcitation Block."""

    def __init__(self, in_channels: int):
        """Initialize."""
        super(SSEBlock, self).__init__()
        self.channel_squeeze = nn.Conv2d(
            in_channels=in_channels, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward."""
        # # x: (bs, ch, h, w) => h: (bs, 1, h, w)
        h = self.sigmoid(self.channel_squeeze(x))
        # # x, h => return: (bs, ch, h, w)
        return x * h
    
    
class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""
    
    def __init__(
        self, in_channels: int,
        out_channels_list: tp.List[int],
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1
        
        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i: i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)
            
        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)
    
    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)
            
        h = h * x
        return h


##################################################
##################    Split data    ##############
##################################################

def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int=42):
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    n_train, n_class = label_arr.shape
    gid_unique = sorted(set(gid_arr))
    n_group = len(gid_unique)

    # # aid_arr: (n_train,), indicates alternative id for group id.
    # # generally, group ids are not 0-index and continuous or not integer.
    gid2aid = dict(zip(gid_unique, range(n_group)))
#     aid2gid = dict(zip(range(n_group), gid_unique))
    aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

    # # count labels by class
    cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

    # # count labels by group id.
    col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
    cnts_by_group = coo_matrix(
        (np.ones(len(label_arr)), (row, col))
    ).dot(coo_matrix(label_arr)).toarray().astype(int)
    del col
    del row
    cnts_by_fold = np.zeros((n_fold, n_class), int)

    groups_by_fold = [[] for fid in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    print("finished preparation", time.time() - start_time)
    for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for fid in range(n_fold):
            # # eval assignment.
            cnts_by_fold[fid] += cnt_by_g
            fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
            cnts_by_fold[fid] -= cnt_by_g

            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fid

        cnts_by_fold[best_fold] += cnt_by_g
        groups_by_fold[best_fold].append(aid)
    print("finished assignment.", time.time() - start_time)

    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]

        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]

        print("[fold {}]".format(fid), end=" ")
        print("n_group: (train, val) = ({}, {})".format(n_group - len(val_groups), len(val_groups)), end=" ")
        print("n_sample: (train, val) = ({}, {})".format(len(train_indexs), len(val_indexs)))

        yield train_indexs, val_indexs


##################################################
################     Albumentation    ############
##################################################


class ImageTransformBase:

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        augmentations_list = [
            self._get_augmentation(aug_name)(**params)
            for aug_name, params in data_augmentations]
        self.data_aug = albumentations.Compose(augmentations_list)

    def __call__(self, pair: tp.Tuple[np.ndarray]) -> tp.Tuple[np.ndarray]:
        """You have to implement this by task"""
        raise NotImplementedError

    def _get_augmentation(self, aug_name: str) -> tp.Tuple[ImageOnlyTransform, DualTransform]:
        """Get augmentations from albumentations"""
        if hasattr(albumentations, aug_name):
            return getattr(albumentations, aug_name)
        else:
            return eval(aug_name)


class ImageTransformForCls(ImageTransformBase):
    """Data Augmentor for Classification Task."""

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        super(ImageTransformForCls, self).__init__(data_augmentations)

    def __call__(self, in_arrs: tp.Tuple[np.ndarray]) -> tp.Tuple[np.ndarray]:
        """Apply Transform."""
        img, label = in_arrs
        augmented = self.data_aug(image=img)
        img = augmented["image"]

        return img, label


##################################################
################       Metrics      ##############
##################################################


class EvalFuncManager(nn.Module):
    """Manager Class for evaluation at the end of epoch"""

    def __init__(
        self,
        iters_per_epoch: int,
        evalfunc_dict: tp.Dict[str, nn.Module],
        prefix: str = "val"
    ) -> None:
        """Initialize"""
        self.tmp_iter = 0
        self.iters_per_epoch = iters_per_epoch
        self.prefix = prefix
        self.metric_names = []
        super(EvalFuncManager, self).__init__()
        for k, v in evalfunc_dict.items():
            setattr(self, k, v)
            self.metric_names.append(k)
        self.reset()

    def reset(self) -> None:
        """Reset State."""
        self.tmp_iter = 0
        for name in self.metric_names:
            getattr(self, name).reset()

    def __call__(self, y: torch.Tensor, t: torch.Tensor) -> None:
        """Forward."""
        for name in self.metric_names:
            getattr(self, name).update(y, t)
        self.tmp_iter += 1

        if self.tmp_iter == self.iters_per_epoch:
            ppe.reporting.report({
                "{}/{}".format(self.prefix, name): getattr(self, name).compute()
                for name in self.metric_names
            })
            self.reset()
            
            
class MeanLoss(nn.Module):
    
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.loss_sum = 0
        self.n_examples = 0
        
    def forward(self, y: torch.Tensor, t: torch.Tensor):
        """Compute metric at once"""
        return self.loss_func(y, t)

    def reset(self):
        """Reset state"""
        self.loss_sum = 0
        self.n_examples = 0
    
    def update(self, y: torch.Tensor, t: torch.Tensor):
        """Update metric by mini batch"""
        self.loss_sum += self(y, t).item() * y.shape[0]
        self.n_examples += y.shape[0]

    def compute(self):
        """Compute metric for dataset"""
        return self.loss_sum / self.n_examples
    

class MyLogLoss(MeanLoss):
    
    def __init__(self, **params):
        super(MyLogLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss(**params)


class MyROCAUC(nn.Module):
    """ROC AUC score"""

    def __init__(self, average="macro") -> None:
        """Initialize."""
        self.average = average
        self._pred_list = []
        self._true_list = []
        super(MyROCAUC, self).__init__()

    def reset(self) -> None:
        """Reset State."""
        self._pred_list = []
        self._true_list = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Forward."""
        self._pred_list.append(y_pred.detach().cpu().numpy())
        self._true_list.append(y_true.detach().cpu().numpy())

    def compute(self) -> float:
        """Calc and return metric value."""
        y_pred = np.concatenate(self._pred_list, axis=0)
        y_true = np.concatenate(self._true_list, axis=0)
        score = roc_auc_score(y_true, y_pred, average=self.average)
        return score

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Forward."""
        self.reset()
        self.update(y_pred, y_true)
        return self.compute()


def set_extensions(
    manager, args, model, device,
    val_loader, optimizer,
    eval_manager, print_progress: bool = False,
):
    """Set extensions for PPE"""
    eval_names = ["val/{}".format(name) for name in eval_manager.metric_names]
    
    log_extentions = [
        ppe_extensions.observe_lr(optimizer=optimizer),
        ppe_extensions.LogReport(),
        # ppe_extensions.PlotReport(["train/loss", "val/loss"], 'epoch', filename='loss.png'),
        # ppe_extensions.PlotReport(["lr"], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            "epoch", "iteration", "lr", "train/loss", *eval_names, "elapsed_time"])
    ]
    if print_progress:
        log_extentions.append(ppe_extensions.ProgressBar(update_interval=20))

    for ext in log_extentions:
        manager.extend(ext)
        
    manager.extend( # evaluation
        ppe_extensions.Evaluator(
            val_loader, model,
            eval_func=lambda *batch: run_eval(args, model, device, batch, eval_manager)),
        trigger=(1, "epoch"))
    
    manager.extend(  # model snapshot
        ppe_extensions.snapshot(target=model, filename="snapshot_epoch_{.epoch}.pth"),
        trigger=ppe.training.triggers.MaxValueTrigger(key="val/metric", trigger=(1, 'epoch')))

    return manager


##################################################
################     Albumentation    ############
##################################################

def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def get_stepper(manager, stgs, scheduler):
    """"""
    def dummy_step():
        pass
    
    def step():
        scheduler.step()
        
    def step_with_epoch_detail():
        scheduler.step(manager.epoch_detail)
        
    
    if stgs["scheduler"]["name"] == None:
        return dummy_step, dummy_step
    
    elif stgs["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
        return dummy_step, step_with_epoch_detail
    
    elif stgs["scheduler"]["name"] == "OneCycleLR":
        return dummy_step, step
    
    else:
        return step, dummy_step



