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


__all__ = ['SimCLR']


class SimCLR(nn.Module):
    """ 
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Implementation: https://github.com/google-research/simclr
    """
    def __init__(self, backbone, feature_size, augmentation, projection_dim=128, temperature=0.5, resized_size=224):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.resized_size = resized_size
        self.backbone = backbone
        self.augmentation = augmentation
        self.projector = Projector(feature_size, hidden_dim=feature_size, out_dim=projection_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        
        if self.augmentation == 'default' or self.augmentation == 'bone_default':
            '''
            if == 'default'
            Then our input would be non bone suppressed images 
            
            if == 'bone_default'
            Then our input would be bone suppressed images, 
            meaning that the bone suppression is our first
            data augmentation before the rest
            '''
            self.augment = self._default_augment_function()
            
        elif self.augmentation == 'bone_supp':
            '''
            1 augmentation will be applied to a non bone suppressed image
            The other augmentation will be applied to a bone suppressed image
            
            Meaning that we will have 1 normal image and 1 bone suppressed
            image as the 2 augmented views
            '''
            self.augment = self._identity_augment_function()
            
        else: #combo
            '''
            Both augmentations will get bone suppressed images 
            and so this acts like bone suppression is 
            the first augmentation that has been applied
            '''
            self.augment = self._bone_supp_combo()
    
    # standard data augmentations used by original SimCLR Authors
    def _default_augment_function(self):
        return T.Compose([
                T.RandomResizedCrop(self.resized_size, scale=(0.08, 1.0)),
                T.RandomHorizontalFlip(), # prob of 0.5
                # colour jittering may not work for grayscale images
                # can only change brightness and contrast like this:
                T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8)], p=0.8), 
                T.RandomApply([T.GaussianBlur(kernel_size=self.resized_size//20*2+1, sigma=(0.1, 2.0))], p=0.5)
                ])
    
    # no transformations applied
    def _identity_augment_function(self):
        return T.Compose([
            T.Resize(self.resized_size) # needed for memory issue
        ])
    
    def _bone_supp_combo(self): # includes bone suppression
        return T.Compose([
            # new combination of augmentations
            T.Resize(self.resized_size), # needed for memory issue
            T.RandomHorizontalFlip(), # prob of 0.5 
            T.RandomApply([T.GaussianBlur(kernel_size=self.resized_size//20*2+1, sigma=(0.1, 2.0))], p=0.5)
        ])
    
    '''
    Can take in 1 or 2 batchs of images and applies the transformation and computes the loss.
    Reason:
    - If we want bone suppression to be a transformation, we would have to input 
      an already bone suppressed image due to the inability of performing the bone
      suppression during pre-training as a transformation like above, caused
      by memory restrictions. Therefore, if you want bone suppression to be 
      applied to 1 view of the image and not the other, then you would pass in
      x = non bone suppressed and y = bone suppressed.
      If you want bone suppression to be applied to both views of the image, 
      then x = bone suppressed and y = None
      If you want no bone suppression to be applied at all, then 
      x = none bone suppressed and y = None
    '''
    def forward(self, x, y=None): 
        if y is not None: 
            x1, x2 = self.augment(x), self.augment(y)
        else:
            x1, x2 = self.augment(x), self.augment(x)
        z1, z2 = self.encoder(x1), self.encoder(x2) 
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss

# normalized temperature-scaled cross entropy loss
def nt_xent_loss(z1, z2, temperature=0.5):
    """ NT-Xent loss """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)
    
    
class Projector(nn.Module):
    """ Projector for SimCLR """
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        
        if hidden_dim is None: # linear projection head
            self.layer1 = nn.Linear(in_dim, out_dim)
        else: # non linear projection head
            self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
    def forward(self, x):
        x = self.layer1(x)
        return x 