import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from osgeo import gdal


""" 
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB")) #np.array for albumentations, it was PIL image
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # convert to grayscale
        mask[mask == 255.0] = 1.0 # sigmoid function will be applied to the final layer, so we need to scale the mask to 0-1
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
 """

class SARDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", ".jpg"))

        ds = gdal.Open(img_path)
        if ds is None:
            print("Failed to open the image")
            sys.exit(1)
        
        band = ds.GetRasterBand(6)
        image = band.ReadAsArray()
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)   
        

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
