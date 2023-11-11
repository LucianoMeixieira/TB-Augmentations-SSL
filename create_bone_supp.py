import os
import datasets
from bone_supp_model import ResNetBSModel
import logging

def create_suppressed_images():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    bone_suppression = ResNetBSModel('CXR-bone-suppression/resnet_bs.h5')

    # read in the TBX11 dataset
    tbx11_dir = os.path.join('DATA', 'TBX11K')

    # currently holding all the train and validation split
    tbx11_trainval = datasets.TBX11Dataset.check_and_load_tbx11_dataset(tbx11_dir=tbx11_dir, split='all_trainval', bone_suppression=bone_suppression)

    # currently holding the test split
    tbx11_test = datasets.TBX11Dataset.check_and_load_tbx11_dataset(tbx11_dir=tbx11_dir, split='all_test', bone_suppression=bone_suppression)
    logging.info("created the TBX11 datasets")
    
    # read in the IEEE dataset
    ieee_dir = os.path.join('DATA', 'IEEE_Dataset')
    ieee_dataset = datasets.IEEEDataset.check_and_load_ieee_dataset(ieee_dir=ieee_dir, bone_suppression=bone_suppression)
    logging.info("created the IEEE dataset")
    
    
if __name__ == "__main__":
    create_suppressed_images()
    logging.info("Finished")