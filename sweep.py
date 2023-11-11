import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only use 1 GPU

import torch
from torch import nn
import torchvision.transforms as T
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
import datasets
from fine_tune_model import ResNetClassifier
import argparse
import optuna
from optuna.trial import TrialState
import logging 
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Performing Hyperparameter Sweep")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("CUDA Device Name: %s", device)

def objective(trial, args): # objective function
    logging.info(f'SSL: {args.ssl}')
    logging.info(f"Augmentation: {args.augmentation}")
    logging.info(f"Sweep: {args.sweep}")
    
    # initialize backbone (resnet50)
    backbone = torchvision.models.resnet50(weights=False)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # adjust ResNet to handle greyscale images 
    feature_size = backbone.fc.in_features # get dimensions of fc layer
    backbone.fc = torch.nn.Identity() # removing the classification head of the model as it will be replaced with projection head

    # define fine tuning models
    model = ResNetClassifier(backbone=backbone, feature_size=feature_size)
    
    # Load the pretrained weights into the backbone
    if args.ssl == "simclr":
        pretrained_weights = torch.load(f'{args.ssl}_pretrained_models/{args.augmentation}/weights/{args.ssl}_epoch_500.pth', map_location='cpu')
        model.backbone.load_state_dict({k.replace('encoder.0.', ''): v for k, v in pretrained_weights.items() if 'encoder.0' in k})
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        lr = trial.suggest_float("lr", 0.001, 0.1)
        epochs = trial.suggest_int("epochs", 80, 150, step=10)
        
    elif args.ssl == "dino":
        pretrained_weights = torch.load(f'{args.ssl}_pretrained_models/{args.augmentation}/weights/{args.ssl}_epoch_100.pth', map_location='cpu')
        model.backbone.load_state_dict({k.replace('teacher.0.', ''): v for k, v in pretrained_weights.items() if 'teacher.0' in k})
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        lr = trial.suggest_float("lr", 0.0001, 0.001)
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        
    elif args.ssl == "swav": 
        pretrained_weights = torch.load(f'{args.ssl}_pretrained_models/{args.augmentation}/weights/{args.ssl}_epoch_200.pth', map_location='cpu')
        model.backbone.load_state_dict({k.replace('encoder.0.', ''): v for k, v in pretrained_weights.items() if 'encoder.0' in k})
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        lr = trial.suggest_float("lr", 0.01, 0.3)
        wd = trial.suggest_float("wd", 0.000001, 0.0001)
        epochs = trial.suggest_int("epochs", 100, 400, step=50)
        
    elif args.ssl == "byol":
        pretrained_weights = torch.load(f'{args.ssl}_pretrained_models/{args.augmentation}/weights/{args.ssl}_epoch_500.pth', map_location='cpu')
        model.backbone.load_state_dict({k.replace('online_encoder.0.', ''): v for k, v in pretrained_weights.items() if 'online_encoder.0' in k})
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        lr = trial.suggest_categorical("lr", [0.4, 0.3, 0.2, 0.1, 0.05, 0.005])
        epochs = trial.suggest_int("epochs", 80, 150, step=10)
    
    else:
        raise SystemExit("Must choose simclr, dino, swav or byol")

    logging.info(f"Loaded in backbone for {args.ssl}")

    if args.sweep == "linear":
        # for linear evaluation, freeze the backbone weights
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # load in the IEEE dataset
    ieee_dir = os.path.join('DATA', 'IEEE_Dataset')
    ieee_dataset = datasets.IEEEDataset.check_and_load_ieee_dataset(ieee_dir=ieee_dir, bone_suppression=None)
    ieee_dataset.sync_shuffle(seed=42)
    
    # Splitting the dataset using sync_split method
    train_dataset, val_dataset, _ = ieee_dataset.sync_split(test_size=0.2, val_size=0.2, seed=42) 
    train_data, train_labels, _ = train_dataset
    val_data, val_labels, _ = val_dataset

    
    val_transform = T.Compose([
        T.ToTensor(),  # Converts the image data to PyTorch tensors.
        T.Lambda(lambda x: x.float()), 
        T.CenterCrop(224) # perform center crop after converted to tensor
    ])
    
    train_dataloader = ieee_dataset.to_torch_dataloader(images=train_data, labels=train_labels, batch_size=batch_size, is_training=True)
    val_dataloader = ieee_dataset.to_torch_dataloader(images=val_data, labels=val_labels, batch_size=batch_size, is_training=False, transform=val_transform)
    
    
    model = model.float().to(device)  # move to GPU

    loss_fn = torch.nn.BCELoss()  # define binary cross entropy loss functions
    
    if args.ssl == "simclr":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr * (math.sqrt(batch_size) / 256), momentum=0.9, weight_decay=0, nesterov=True)  # define Nesterov momentum optimizers
        scheduler = None
        
    elif args.ssl == "dino":
        optimizer = torch.optim.SGD(model.parameters(), lr=(lr * (batch_size / 256)), momentum=0.9, weight_decay=0)
        scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0)
    
    elif args.ssl == "swav":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0)
    
    elif args.ssl == "byol":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True) # define Nesterov momentum optimizers
        scheduler = None
    
    else:
        raise SystemExit("Must choose simclr, dino, swav ot byol")

    logging.info("Entering Training Loop")
    best_val_loss = float('inf')  # Initialize with a high value for best val loss
    for epoch in range(epochs):
        model.train()
        for batch_1, labels_1 in train_dataloader:
            batch_1, labels_1 = batch_1.float().to(device), labels_1.float().to(device)  # move to GPU
            train_prediction = model(batch_1).squeeze(1)  # forward pass --> gets probability from Sigmoid
            loss_1 = loss_fn(train_prediction, labels_1)  # get loss --> takes in probabilities

            optimizer.zero_grad()  # zero out gradients
            loss_1.backward()  # backprop and update gradients
            optimizer.step()
        
        if scheduler:
            scheduler.step()

        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():  # dont update gradients for validation
            for batch_2, labels_2 in val_dataloader:
                batch_2, labels_2 = batch_2.float().to(device), labels_2.float().to(device)
                val_prediction = model(batch_2, eval=True).squeeze(1)  # forward pass --> gets probability from Sigmoid
                val_loss = loss_fn(val_prediction, labels_2) # get loss --> takes in probabilities
                total_val_loss += val_loss.item()

        val_epoch_loss = total_val_loss / len(val_dataloader)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
        
        trial.report(val_epoch_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for different sweeps.")
    
    # Define the mode argument
    parser.add_argument("ssl", choices=["simclr", "dino", "swav", "byol"], 
                        help= "SSL technique to run.")

    # Define the augmentation argument
    parser.add_argument("augmentation", choices=["default", "bone_supp", "bone_default", "combo"], 
                        help="Augmentation stategy to use.")
    
    # Define the sweep argument
    parser.add_argument("sweep", choices=["linear", "full"], 
                        help="Type of sweep to conduct.")

    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize") # minimize validation loss
    study.optimize(lambda trial: objective(trial, args), n_trials=20)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logging.info("Study statistics:")
    logging.info("  Number of finished trials: %d", len(study.trials))
    logging.info("  Number of pruned trials: %d", len(pruned_trials))
    logging.info("  Number of complete trials: %d", len(complete_trials))

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: %f", trial.value)

    logging.info("  Params:")
    for key, value in trial.params.items():
        logging.info("    %s: %s", key, value)

    logging.info("Finished")