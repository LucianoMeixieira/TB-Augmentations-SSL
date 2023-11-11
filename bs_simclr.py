'''LARS optimizer from this repo: https://github.com/kakaobrain/torchlars/tree/master'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only use 1 GPU

import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import datasets
from SSL import SimCLR
from fine_tune_model import ResNetClassifier
from torchlars import LARS # error as need linux and CUDA but only available on the cluster
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import logging 
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, auc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

BATCH_SIZE = 256 
EPOCHS_PRE = 500
WARM_UP = 10 # epochs for linear warm up 
LEARNING_RATE_PRE = 0.075 * math.sqrt(BATCH_SIZE) # Square root LR scaling works better for smaller batch size trained in fewer epochs
MOMENTUM = 0.9
WEIGHT_DECAY = 10e-6
SEED = 42

def main(args):
    # Ensuring that CUDA is enabled for GPU utilization
    logging.info("PyTorch version: %s", torch.__version__)
    logging.info("CUDA available: %s", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Exiting...")

    logging.info("Number of CUDA devices: %s", torch.cuda.device_count())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("CUDA Device Name: %s", device)

    logging.info(f'Mode: {args.mode}')
    logging.info(f"Augmentation: {args.augmentation}")
    
    # initialize backbone (resnet50)
    backbone = torchvision.models.resnet50(weights=False)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # adjust ResNet to handle greyscale images 
    feature_size = backbone.fc.in_features # get dimensions of fc layer
    backbone.fc = torch.nn.Identity() # removing the classification head of the model as it will be replaced with projection head
    
    # --------------------------------------------pretraining---------------------------------------------------
    if args.mode == 'pretrain': 
        
        # Define a main directory for pretrained models
        BASE_DIR = 'simclr_pretrained_models'
        SAVE_DIR = os.path.join(BASE_DIR, args.augmentation)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        # Define a directory to save the model loss
        LOSS_DIR = os.path.join(SAVE_DIR, 'loss')
        if not os.path.exists(LOSS_DIR):
            os.makedirs(LOSS_DIR)
            
        # Define a directory to save the model weights
        WEIGHTS_DIR = os.path.join(SAVE_DIR, 'weights')
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
                
        # read in the TBX11 dataset
        tbx11_dir = os.path.join('DATA', 'TBX11K')

        # currently holding all the train and validation split
        tbx11_trainval = datasets.TBX11Dataset.check_and_load_tbx11_dataset(tbx11_dir=tbx11_dir, split='all_trainval', bone_suppression=None)
        
        # currently holding the test split
        tbx11_test = datasets.TBX11Dataset.check_and_load_tbx11_dataset(tbx11_dir=tbx11_dir, split='all_test', bone_suppression=None)
        
        # creating the full dataset to be used for pretraining as we don't need splits
        tbx11_trainval.merge_with(tbx11_test)
        logging.info(f'Loaded in {len(tbx11_trainval.data)} images and {len(tbx11_trainval.bone_supp_data)} bone suppressed images')

        if len(tbx11_trainval.bone_supp_data) == 0:
            raise SystemExit("No bone suppressed images detected. Please run the create datasets script")
        
        # create SimCLR model
        simclr = SimCLR(backbone=backbone, feature_size=feature_size, augmentation=args.augmentation)
        logging.info(f'Created {args.augmentation} SimCLR Model')
        
        # pretrain the SimCLR model
        loss = pretrain(
            model=simclr, 
            augmentation=args.augmentation, 
            device=device, 
            dataset=tbx11_trainval, 
            WEIGHTS_DIR=WEIGHTS_DIR
        )
        logging.info('Pretrained the model')    
        save_loss(loss, LOSS_DIR, 'SimCLR Pre-Training Loss')
    
    else: # either fine_tuning or evaluating
        
        # define directories
        BASE_DIR = "simclr_fine_tuned_models"
        SAVE_DIR = os.path.join(BASE_DIR, args.augmentation)
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        # Define a directory to save the model loss
        LOSS_DIR = os.path.join(SAVE_DIR, 'loss')
        if not os.path.exists(LOSS_DIR):
            os.makedirs(LOSS_DIR)
            
        # Define a directory to save the model weights
        WEIGHTS_DIR = os.path.join(SAVE_DIR, 'weights')
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
        
        # define a directory to save the evaluation results
        EVALS_DIR = os.path.join(SAVE_DIR, 'evals')
        if not os.path.exists(EVALS_DIR):
            os.makedirs(EVALS_DIR)
        
        
        # load in the IEEE dataset
        ieee_dir = os.path.join('DATA', 'IEEE_Dataset')
        ieee_dataset = datasets.IEEEDataset.check_and_load_ieee_dataset(ieee_dir=ieee_dir, bone_suppression=None)
        ieee_dataset.sync_shuffle(seed=SEED)
        
        # Splitting the dataset using sync_split method
        train_dataset, val_dataset, test_dataset = ieee_dataset.sync_split(test_size=0.2, val_size=0.1, seed=SEED) 
        train_data, train_labels, _ = train_dataset # dont need bone supp data and so _
        val_data, val_labels, _ = val_dataset
        test_data, test_labels, _ = test_dataset
            
        # --------------------------------------------fine-tuning---------------------------------------------------
        if args.mode == 'train':
            
            # read in pretrained models
            pretrained_weights = torch.load(f'simclr_pretrained_models/{args.augmentation}/weights/simclr_epoch_500.pth', map_location='cpu')
            
            # define fine tuning models
            linear_eval = ResNetClassifier(backbone=backbone, feature_size=feature_size)
            full_eval = ResNetClassifier(backbone=backbone, feature_size=feature_size)
            
            # Load the pretrained SimCLR weights into the backbone
            linear_eval.backbone.load_state_dict({k.replace('encoder.0.', ''): v for k, v in pretrained_weights.items() if 'encoder.0' in k})
            full_eval.backbone.load_state_dict({k.replace('encoder.0.', ''): v for k, v in pretrained_weights.items() if 'encoder.0' in k})
            
            
            logging.info("Models have been loaded in")
            
            # for linear evaluation, freeze the backbone weights
            for param in linear_eval.backbone.parameters():
                param.requires_grad = False
            
            val_transform = T.Compose([
                T.ToTensor(),  # Converts the image data to PyTorch tensors.
                T.Lambda(lambda x: x.float()), 
                T.CenterCrop(224) # perform center crop after converted to tensor
            ])
            
            full_config = {
                'batch_size': 256, 
                'lr': 0.01, 
                'epochs': 110
            }
            
            linear_config = {
                'batch_size': 256, 
                'lr': 0.08, 
                'epochs': 140
            }
            
            
            train_dataloader = ieee_dataset.to_torch_dataloader(images=train_data, labels=train_labels, batch_size=BATCH_SIZE, is_training=True)
            val_dataloader = ieee_dataset.to_torch_dataloader(images=val_data, labels=val_labels, batch_size=BATCH_SIZE, is_training=False, transform=val_transform)
            
            train_loss_full, val_loss_full = fine_tune(full_eval, device, full_config, train_dataloader, val_dataloader, "full", WEIGHTS_DIR)
            logging.info("Full Fine Tuning Complete")
            save_fine_tune_loss(train_loss_full, val_loss_full, LOSS_DIR, "full")
            
            train_loss_linear, val_loss_linear = fine_tune(linear_eval, device, linear_config, train_dataloader, val_dataloader, "linear", WEIGHTS_DIR)
            logging.info("Linear Fine Tuning Complete")
            save_fine_tune_loss(train_loss_linear, val_loss_linear, LOSS_DIR, "linear")
            
        # --------------------------------------------evaluating---------------------------------------------------
        else: 
            
            test_dataloader = ieee_dataset.to_torch_dataloader(images=test_data, labels=test_labels, batch_size=BATCH_SIZE, is_training=False)
            
            linear_weights = torch.load(f'simclr_fine_tuned_models/{args.augmentation}/weights/simclr_fine_tuned_linear.pth', map_location='cpu')
            full_weights = torch.load(f'simclr_fine_tuned_models/{args.augmentation}/weights/simclr_fine_tuned_full.pth', map_location='cpu')
            
            linear_trained = ResNetClassifier(backbone=backbone, feature_size=feature_size)
            full_trained = ResNetClassifier(backbone=backbone, feature_size=feature_size)
            
            linear_trained.load_state_dict(linear_weights)
            full_trained.load_state_dict(full_weights)
            
            logging.info("Models Have Been Loaded In")
            
            linear_filename = os.path.join(EVALS_DIR, f"linear_metrics_{args.augmentation}.txt")
            full_filename = os.path.join(EVALS_DIR, f"full_metrics_{args.augmentation}.txt")
            
            labels_linear, predicts_linear = eval(linear_trained, test_dataloader, device, linear_filename)
            linear_save_cm = os.path.join(EVALS_DIR, f"linear_conf_{args.augmentation}.png")
            linear_save_roc = os.path.join(EVALS_DIR, f"linear_roc_{args.augmentation}.png")
            plot_confusion_matrix(labels_linear, predicts_linear, classes=[0, 1], filename=linear_save_cm, augment=args.augmentation)
            plot_auc_roc(labels_linear, predicts_linear, augment=args.augmentation, save_path=linear_save_roc)
            logging.info("Evaluted the Linear Model")
            
            labels_full, predicts_full = eval(full_trained, test_dataloader, device, full_filename)
            full_save_cm = os.path.join(EVALS_DIR, f"full_conf_{args.augmentation}.png")
            full_save_roc = os.path.join(EVALS_DIR, f"full_roc_{args.augmentation}.png")
            plot_confusion_matrix(labels_full, predicts_full, classes=[0, 1], filename=full_save_cm, augment=args.augmentation)
            plot_auc_roc(labels_full, predicts_full, augment=args.augmentation, save_path=full_save_roc)
            logging.info("Evaluted the Full Model")
            

def pretrain(model, augmentation, device, dataset, WEIGHTS_DIR):
    # enable CUDA
    model = model.float().to(device) 
    
    # Base optimizer
    base_optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_PRE, weight_decay=WEIGHT_DECAY)
    
    # Using the LARS optimizer from torchlars
    optimizer = LARS(optimizer=base_optimizer) 
    
    # Cosine decay scheduler after linear warm up
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(EPOCHS_PRE - WARM_UP))
    loss_values = []
    
    # get in training mode
    model.train() 
    
    logging.info("Entering the training loop")
    for epoch in range(EPOCHS_PRE):
        # shuffle the normal images and bone suppressed images, but they still correspond
        dataset.sync_shuffle(seed=epoch)
        
        if augmentation == 'bone_supp': # if bone supp, need bone supp image and normal image
            data1 = dataset.data
            data2 = dataset.bone_supp_data
            
        else: # if default only need normal images. If bone_default or combo only need bone supp images
            data1 = dataset.data if augmentation == 'default' else dataset.bone_supp_data
            data2 = None
        
        # train the model for 1 epoch and get the loss from simclr
        epoch_loss = pretrain_epoch(model, optimizer, dataset, data1, data2, epoch, cosine_scheduler, device)
        loss_values.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0: # only save the weights every 10 epochs
            logging.info(f'Epoch {epoch + 1}, Average Loss: {epoch_loss:.4f}')
            
            save_path = os.path.join(WEIGHTS_DIR, f'simclr_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'Model saved to {save_path}')
    
    return loss_values

def pretrain_epoch(model, optimizer, dataset, data1, data2, epoch, cosine_scheduler, device):
    total_loss = 0.0
    dataloader1 = dataset.to_torch_dataloader(images=data1, batch_size=BATCH_SIZE)
    
    if data2:
        dataloader2 = dataset.to_torch_dataloader(images=data2, batch_size=BATCH_SIZE)
        for norm_batch, sup_batch in zip(dataloader1, dataloader2):
            norm_batch, sup_batch = norm_batch.float().to(device), sup_batch.float().to(device) # enable CUDA
            optimizer.zero_grad() # zero the parameter gradients
            loss = model(norm_batch, sup_batch) # compute loss --> returned from forward function
            total_loss += loss.item()
            
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
    else:
        for batch in dataloader1:
            batch = batch.float().to(device) # enable CUDA
            optimizer.zero_grad() # zero the parameter gradients
            loss = model(batch) # compute loss --> returned from forward function
            total_loss += loss.item()

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            
    # Linear warm-up for the first 10 epochs
    if epoch < WARM_UP:
        lr_scale = (epoch + 1) / WARM_UP
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE_PRE * lr_scale
            
    # Cosine decay after 10 epochs
    else:
        cosine_scheduler.step()
    
    return total_loss / len(dataloader1)

def fine_tune(model, device, config, train_dataloader, val_dataloader, sweep, WEIGHTS_DIR):
    model = model.float().to(device)
    loss_fn = torch.nn.BCELoss()  # define binary cross entropy loss functions
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'] * (math.sqrt(config['batch_size']) / 256), momentum=MOMENTUM, weight_decay=0, nesterov=True)  # define Nesterov momentum optimizers
    training_loss, validation_loss = [], []
    
    logging.info("Entering Training Loop")
    for epoch in range(config['epochs']):
        total_train_loss = 0.0
        model.train()
        for batch_1, labels_1 in train_dataloader:
            batch_1, labels_1 = batch_1.float().to(device), labels_1.float().to(device)  # move to GPU
            train_prediction = model(batch_1).squeeze(1)  # forward pass --> gets probability from Sigmoid
            train_loss = loss_fn(train_prediction, labels_1)  # get loss --> takes in probabilities
            total_train_loss += train_loss.item()
            
            optimizer.zero_grad()  # zero out gradients
            train_loss.backward()  # backprop and update gradients
            optimizer.step()
        
        train_epoch_loss = total_train_loss / len(train_dataloader)
        training_loss.append(train_epoch_loss)
        
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():  # dont update gradients for validation
            for batch_2, labels_2 in val_dataloader:
                batch_2, labels_2 = batch_2.float().to(device), labels_2.float().to(device)
                val_prediction = model(batch_2, eval=True).squeeze(1)  # forward pass --> gets probability from Sigmoid
                val_loss = loss_fn(val_prediction, labels_2) # get loss --> takes in probabilities
                total_val_loss += val_loss.item()
                
        val_epoch_loss = total_val_loss / len(val_dataloader)
        validation_loss.append(val_epoch_loss)
        
        logging.info(f'Epoch {epoch + 1}, Training Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')
    
    # then save models weights
    save_path = os.path.join(WEIGHTS_DIR, f'simclr_fine_tuned_{sweep}.pth')
    torch.save(model.state_dict(), save_path)
    logging.info(f'Model saved to {save_path}')
    
    return training_loss, validation_loss
    
def save_loss_curve(train_loss, path, title, val_loss=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, 'b', label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(path)

    # close the figure to free up memory
    plt.close()

def save_loss(loss, l_dir, title):
    # Save the loss values to a .npy file
    loss_values_path = os.path.join(l_dir, 'loss_values.npy')
    np.save(loss_values_path, loss)
    
    # Save the loss curve
    loss_curve_path = os.path.join(l_dir, 'loss_curve.png')
    save_loss_curve(loss, loss_curve_path, title)
    logging.info('Saved the loss')
    
def save_fine_tune_loss(train_loss, val_loss, l_dir, sweep):
    # Save the loss values to a .npy file
    train_loss_values_path = os.path.join(l_dir, f'train_loss_values_{sweep}.npy')
    np.save(train_loss_values_path, train_loss)
    
    val_loss_values_path = os.path.join(l_dir, f'val_loss_values_{sweep}.npy')
    np.save(val_loss_values_path, val_loss)
    
    # Save the loss curve
    loss_curve_path = os.path.join(l_dir, f'loss_curve_{sweep}.png')
    save_loss_curve(train_loss, loss_curve_path, f"SimCLR {sweep} sweep loss curves", val_loss)
    logging.info('Saved the loss curve')
    
def eval(model, dataloader, device, filename):
    logging.info("Evaluating Model")
    all_predicts, all_labels = [], []
    model = model.to(device).float()
    model.eval() # put in evaluation mode
    with torch.no_grad():
        for batch, labels in dataloader:
            batch, labels = batch.float().to(device), labels.float().to(device)
            prob = model(batch, eval=True).squeeze(1)  # forward pass --> gets probability from Sigmoid
            predicted = (prob > 0.5).float() # Convert model's probability output to binary labels 0 or 1
            
            all_predicts.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logging.info("Calculating Metrics")
    conf_matrix = confusion_matrix(all_labels, all_predicts)
    precision = precision_score(all_labels, all_predicts)
    recall = recall_score(all_labels, all_predicts)
    accuracy = accuracy_score(all_labels, all_predicts)
    roc_auc = roc_auc_score(all_labels, all_predicts)
    tn, fp, _, _ = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    class_report = classification_report(all_labels, all_predicts)

    with open(filename, 'w') as file:
        file.write("Confusion Matrix:\n")
        file.write(str(conf_matrix))
        file.write("\n")
        
        file.write("Accuracy: " + str(accuracy) + "\n")
        file.write("Precision: " + str(precision) + "\n")
        file.write("Sensitivity (Recall): " + str(recall) + "\n")
        file.write("Specificity: " + str(specificity) + "\n")
        file.write("ROC-AUC: " + str(roc_auc) + "\n")
        
        file.write("\nClassification Report:\n")
        file.write(class_report)
        
    return all_labels, all_predicts

def plot_confusion_matrix(true_labels, predicted_labels, classes, filename, augment, 
                    figsize=(10,7), fontsize=14):

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(classes)))
    
    # Create a DataFrame for easier slicing and relabeling
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
    
    # Set labels and title
    ax.set_ylabel('Actual', fontsize=fontsize)
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_title(f'Confusion Matrix For SimCLR {augment}', fontsize=fontsize+2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    # Save the figure
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    
def plot_auc_roc(true_labels, predicted_labels, augment, save_path):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for SimCLR {augment}')
    plt.legend(loc="lower right")
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for different modes and augmentations.")
    
    # Define the mode argument
    parser.add_argument("mode", choices=["pretrain", "train", "evaluate"], 
                        help= "Mode to run the script in.")

    # Define the augmentation argument
    parser.add_argument("augmentation", choices=["default", "bone_supp", "bone_default", "combo"], 
                        help="Augmentation stategy to use'.")

    args = parser.parse_args()
    logging.info("Entering Main")
    main(args)
    logging.info("Finished")