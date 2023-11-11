import torch.nn as nn
import torchvision.transforms as T

class ResNetClassifier(nn.Module):
    def __init__(self, backbone, feature_size, resized_size=224):
        super(ResNetClassifier, self).__init__()
        self.backbone = backbone # learned backbone from SSL
        self.classifier = nn.Sequential( # classification head which replaces the SimCLR Projection head
            nn.Linear(feature_size, 1),
            nn.Sigmoid() # outputs a prob between 0-1
        )
        # Initialize weights and biases
        self.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
        self.classifier[0].bias.data.zero_()
        
        self.resized_size = resized_size
        self.augment = self._augment()
    
    # NOTE, only use this for fine tune training, not validation or evaluation. Evaluation = NO TRANSFORMATION
    def _augment(self): 
        return T.Compose([
            T.RandomResizedCrop(self.resized_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(), # prob of 0.5
            ])

    def forward(self, x, eval=False):
        if not eval: # training
            x_aug = self.augment(x)
            features = self.backbone(x_aug)
            
        else: # validation or evaluation
            features = self.backbone(x)

        out = self.classifier(features)
        return out