import os
import numpy as np
from skimage.util import img_as_float
from sklearn.model_selection import train_test_split
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

def preprocess_images(img):
    img = cv2.resize(img, (256, 256)) # Resize image to (256, 256)
    img = img_as_float(img)  # Convert to float
    return img

# Define a custom X_Rays class for Pytorch Dataloader to handle 
# grayscale images and their corresponding labels if provided.
class X_Rays(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data # List of preprocessed images, either bone suppressed or not
        self.labels = labels # corresponding labels 
        self.transform = transform # optional transformations to be applied

    # number of samples in the dataset
    def __len__(self): 
        return len(self.data)

    # get sample at specified index
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx] # for IEEE
        else:
            return sample # For TBX11

# This dataset will be used for fine tuning and testing
class IEEEDataset:
    def __init__(self, root_dir, bone_suppression_model):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.bone_supp_data = [] # will hold the bone suppressed versions of the data
        self.bone_suppression_model = bone_suppression_model  # Assign the model to a class variable
        self._load_data()

    # loads in all our image and label data
    def _load_data(self):
        normal_dir = os.path.join(self.root_dir, 'Normal')
        tb_dir = os.path.join(self.root_dir, 'Tuberculosis')
        i, j = 0, 0
        logging.info("Creating the IEEE Dataset")
        
        # Read and preprocess the 'Normal' images and assign label 0
        for filename in os.listdir(normal_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(normal_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read in as grayscale
                if img is not None:
                    img = preprocess_images(img)
                    self.data.append(img)
                    self.labels.append(0)
                    if self.bone_suppression_model is not None:
                        self.bone_supp_data.append(self.bone_suppression_model.predict(img, verbose=0))
                    
                    if i % 1000 == 0:
                        logging.info(f'Loaded in {i} number of images')
                    i += 1
                    
                else:
                    logging.info(f"Unable to read {img_path}")

        # Read and preprocess the 'TB' images and assign label 1
        for filename in os.listdir(tb_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(tb_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read in as grayscale
                if img is not None:
                    img = preprocess_images(img)
                    self.data.append(img)
                    self.labels.append(1)
                    if self.bone_suppression_model is not None:
                        self.bone_supp_data.append(self.bone_suppression_model.predict(img, verbose=0))
                    
                    if j % 1000 == 0:
                        logging.info(f'Loaded in {j} number of images')
                    j += 1
                    
                else:
                    logging.info(f"Unable to read {img_path}")

        logging.info(f"Read in {len(self.data)} images and made {len(self.bone_supp_data)} bone suppression images with {len(self.labels)} labels")
        
        
    # Shuffle the data, labels, and bone_supp_data in sync to keep the correspondence between them
    def sync_shuffle(self, seed=None):
        indices = np.arange(len(self.data))
        np.random.seed(seed)
        np.random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.bone_supp_data = [self.bone_supp_data[i] for i in indices]

    # Split the dataset into train, validation, and test sets in sync to keep the correspondence between them
    def sync_split(self, test_size=0.2, val_size=0.2, seed=None):        
        train_indices, temp_indices = train_test_split(np.arange(len(self.data)), test_size=test_size + val_size, random_state=seed)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (test_size + val_size), random_state=seed)

        # these are tuples which have the following form:
        # (subset_data, subset_labels, subset_bone_supp_data)
        train_dataset = self._subset(train_indices)
        val_dataset = self._subset(val_indices)
        test_dataset = self._subset(test_indices)

        return train_dataset, val_dataset, test_dataset

    # Create a subset of data, labels, and bone_supp_data using the provided indices
    def _subset(self, indices):
        subset_data = [self.data[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        subset_bone_supp_data = [self.bone_supp_data[i] for i in indices]

        return subset_data, subset_labels, subset_bone_supp_data # returned as a tuple
    
    # Converts the data to a PyTorch DataLoader object.
    def to_torch_dataloader(self, images, labels, batch_size, is_training=True, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts the image data to PyTorch tensors.
                transforms.Lambda(lambda x: x.float())
            ])
        
        dataset = X_Rays(data=images, labels=labels, transform=transform)
        
        # Creating a DataLoader with shuffling for training datasets and without shuffling for validation/test datasets.
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=is_training, 
                                pin_memory=True)
        return dataloader
    
    '''
    Here cls refers to the class itself, and cls(root_dir=ieee_dir, bone_suppression_model=None) 
    is equivalent to calling IEEEDataset(root_dir=ieee_dir, bone_suppression_model=None), 
    which invokes the __init__ method and returns a new instance of the class.
    '''
    @classmethod
    def check_and_load_ieee_dataset(cls, ieee_dir, bone_suppression):
        bone_supp_dir = 'IEEE_bone_supp'

        # Check if ieee_dir exists
        if not os.path.exists(ieee_dir):
            logging.info(f"{ieee_dir} does not exist.")
            raise FileNotFoundError(f"{ieee_dir} does not exist.")
        
        # if the bone supp data folder exists
        if os.path.exists(bone_supp_dir):
            logging.info("Folder Exists")
            ieee_dataset = cls(root_dir=ieee_dir, bone_suppression_model=None)

            # Load the bone_supp_data from the IEEE_bone_supp folder
            # File names are sorted by their indices to maintain order
            file_names = sorted([file_name for file_name in os.listdir(bone_supp_dir) if file_name.endswith('.npy')],
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))
            ieee_dataset.bone_supp_data = [np.load(os.path.join(bone_supp_dir, file_name)) for file_name in file_names]

        else:
            logging.info("Folder Does Not Exist")
            ieee_dataset = cls(root_dir=ieee_dir, bone_suppression_model=bone_suppression)

            if bone_suppression is not None:
                # Create the bone_supp folder and save the bone_supp_data into it
                os.makedirs(bone_supp_dir)
                for idx, image in enumerate(ieee_dataset.bone_supp_data):
                    file_path = os.path.join(bone_supp_dir, f'ieee_bone_supp_{idx:03d}.npy')  # Added zero padding to index
                    np.save(file_path, image)

        return ieee_dataset
        

# for this dataset, we don't need labels as we will be using this for SSL
class TBX11Dataset:
    def __init__(self, root_dir, split, bone_suppression_model):
        self.root_dir = root_dir
        self.split = split
        self.imgs_dir = os.path.join(root_dir, 'imgs')
        self.lists_dir = os.path.join(root_dir, 'lists')
        self.data = []
        self.bone_supp_data = [] # will hold the bone suppressed versions of the data
        self.bone_suppression_model = bone_suppression_model  # Assign the model to a class variable
        self._load_data()
    
    # loads in all the image data
    def _load_data(self):
        list_file_name = f'{self.split}.txt' # read in the list txt file
        list_file_path = os.path.join(self.lists_dir, list_file_name)
        i = 0
        logging.info("Creating the TBX11 Dataset")
        
        with open(list_file_path, 'r') as file:
            for line in file:
                img_rel_path = line.strip()  # Each line is a relative path to an image from 'imgs' directory.
                img_path = os.path.join(self.imgs_dir, img_rel_path)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read in as grayscale
                if img is not None:
                    img = preprocess_images(img) # pre-process the images
                    self.data.append(img)
                    if self.bone_suppression_model is not None:
                        self.bone_supp_data.append(self.bone_suppression_model.predict(img, verbose=0))
                        
                    if i % 1000 == 0:
                        logging.info(f'Loaded in {i} number of images')
                    i += 1
                    
                else:
                    logging.info(f"Unable to read {img_path}")
                    
        logging.info(f"Read in {len(self.data)} images and made {len(self.bone_supp_data)} bone suppression images from split: {self.split}")
    
    # Converts the data to a PyTorch DataLoader object.
    def to_torch_dataloader(self, images, *_, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image data to PyTorch tensors.
            transforms.Lambda(lambda x: x.float())
        ])
        
        dataset = X_Rays(data=images, labels=None, transform=transform)
        
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                pin_memory=True)
        return dataloader
    
    # Shuffle the data, labels, and bone_supp_data in sync to keep the correspondence between them
    def sync_shuffle(self, seed=None):
        indices = np.arange(len(self.data))
        np.random.seed(seed)
        np.random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.bone_supp_data = [self.bone_supp_data[i] for i in indices]
    
    def merge_with(self, other_dataset):
        if not isinstance(other_dataset, TBX11Dataset):
            raise ValueError("Input should be an instance of TBX11Dataset")
        
        # Extend the data and bone_supp_data attributes of self
        self.data.extend(other_dataset.data)
        self.bone_supp_data.extend(other_dataset.bone_supp_data)
        
        # Set the split attribute to "merged"
        self.split = "merged"
        
        return self
        
    @classmethod
    def check_and_load_tbx11_dataset(cls, tbx11_dir, split, bone_suppression):
        bone_supp_dir = f'TBX11_bone_supp_{split}'
        
        # Check if tbx11_dir exists
        if not os.path.exists(tbx11_dir):
            logging.info(f"{tbx11_dir} does not exist.")
            raise FileNotFoundError(f"{tbx11_dir} does not exist.")

        # if the bone supp data folder exists
        if os.path.exists(bone_supp_dir):
            logging.info("Folder Exists")
            tbx11_dataset = cls(root_dir=tbx11_dir, split=split, bone_suppression_model=None)

            # Load the bone_supp_data from the bone_supp folder
            # File names are sorted by their indices to maintain order
            file_names = sorted([file_name for file_name in os.listdir(bone_supp_dir) if file_name.endswith('.npy')],
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))
            tbx11_dataset.bone_supp_data = [np.load(os.path.join(bone_supp_dir, file_name)) for file_name in file_names]

        else:
            logging.info("Folder Does Not Exist")
            tbx11_dataset = cls(root_dir=tbx11_dir, split=split, bone_suppression_model=bone_suppression)

            if bone_suppression is not None:
                # Create the bone_supp folder and save the bone_supp_data into it
                os.makedirs(bone_supp_dir)
                for idx, image in enumerate(tbx11_dataset.bone_supp_data):
                    file_path = os.path.join(bone_supp_dir, f'tbx11_bone_supp_{split}_{idx:03d}.npy')  # Added zero padding to index
                    np.save(file_path, image)

        return tbx11_dataset