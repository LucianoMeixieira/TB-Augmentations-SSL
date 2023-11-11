# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
This code was adapted from the following repository: "https://github.com/giakou4/pyssl":
'''

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import copy
from PIL import Image


__all__ = ['BYOL']


class BYOL(nn.Module):
    """ 
    BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
    Link: https://arxiv.org/abs/2006.07733
    Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol
    """
    def __init__(self, backbone, feature_size, augmentation, device, projection_dim=256, hidden_dim=4096, tau=0.9995,
                 resized_size=224):
        super().__init__()
        self.projection_dim = projection_dim
        self.tau = tau # EMA update
        self.backbone = backbone
        self.projector = MLP(feature_size, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.resized_size = resized_size
        self.augmentation = augmentation
        self.online_encoder =  self.encoder = nn.Sequential(self.backbone, self.projector)
        self.online_predictor = MLP(in_dim=projection_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder) # target must be a deepcopy of online, since we will use the backbone trained by online
        self._init_target_encoder()
        self.augment1 = self._augment1()
        self.augment2 = self._augment2()
        self.accumulated_gradients = {}
        for name, param in self.online_encoder.named_parameters():
            self.accumulated_gradients[name] = torch.zeros_like(param.data).to(device)
    
    def _augment1(self):
        if self.augmentation == "default" or self.augmentation == "bone_default":
            return T.Compose([
                    T.RandomResizedCrop(self.resized_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ])
        
        elif self.augmentation == "bone_supp":
            return T.Compose([
                    T.Resize(self.resized_size) # will be non bone suppressed
                ])
            
        else: #combo 
            return T.Compose([
                    T.Resize(self.resized_size), # will be non bone suppressed
                    T.RandomHorizontalFlip(p=0.5),
                    T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ])
            
    def _augment2(self):
        if self.augmentation == "default" or self.augmentation == "bone_default":
            return T.Compose([
                    T.RandomResizedCrop(self.resized_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                    T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
                    T.RandomSolarize(threshold=0.5, p=0.2)
                ])
            
        elif self.augmentation == "bone_supp":
            return T.Compose([
                    T.Resize(self.resized_size) # will be non bone suppressed
                ])
        
        else: #combo
            return T.Compose([
                    T.Resize(self.resized_size), # will be non bone suppressed
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
                    T.RandomSolarize(threshold=0.5, p=0.2)
                ])
        
    def forward(self, x, y = None):
        if y is not None:
            x1, x2 = self.augment1(x), self.augment2(y)
            
        else:
            x1, x2 = self.augment1(x), self.augment2(x)
            
        z1_o, z2_o = self.online_encoder(x1), self.online_encoder(x2)
        p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
        with torch.no_grad():
            z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2) 
        loss =  mean_squared_error(p1_o, z2_t) / 2 + mean_squared_error(p2_o, z1_t) / 2     
        return loss
    
    def _init_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
            
    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.tau * param_t.data  + (1. - self.tau) * param_o.data
            
    @torch.no_grad()
    def apply_accumulated_gradients(self, N):
        for name, param in self.online_encoder.named_parameters():
            if name in self.accumulated_gradients:
                avg_grad = self.accumulated_gradients[name] / N
                param.grad = avg_grad.clone()
                self.accumulated_gradients[name].zero_()
                  

def mean_squared_error(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


class MLP(nn.Module):
    """ Projection Head and Prediction Head for BYOL """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
    