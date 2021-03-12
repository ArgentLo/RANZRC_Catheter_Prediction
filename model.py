import torch
import torch.nn as nn
import timm
from warnings import filterwarnings
filterwarnings("ignore")


class REXNET_100(nn.Module):
    def __init__(self, model_name='rexnet_100', out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # print(self.model)
        
        ## pretrain PATH on Kaggle
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.head.fc.in_features
        self.model.head.global_pool = nn.Identity()
        self.model.head.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class RESNEST_50D(nn.Module):
    def __init__(self, model_name='resnest50d', out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output



class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        ## pretrain PATH on Kaggle
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


class EffNet_b3(nn.Module):
    def __init__(self, model_name="efficientnet_b3", out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # print(self.model)
        ## pretrain PATH on Kaggle
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class EffNet_b0(nn.Module):
    def __init__(self, model_name="efficientnet_b0", out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # print(self.model)
        ## pretrain PATH on Kaggle
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output