import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class mri_data(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.imgs = 1 #img list
        
    def __getitem__(self, index):
        img = Image.open(
            r'D:\python\Unet\project\Image-segmentation\test.tif').convert('RGB') #w, h = 256
        
        if self.transform is not None:
            img = self.transform(img)
        # img = img.unsqueeze(0)
        return img
        
    def __len__(self):
        return 1

def train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

def get_data():
    train_set = mri_data(transform = train_transform())

    train_loader = DataLoader(
        dataset=train_set, batch_size=1, shuffle=False, num_workers=0)
    for img in train_loader:
        return img

if __name__ == '__main__':
    train_set = mri_data(transform = train_transform())

    train_loader = DataLoader(
        dataset=train_set, batch_size=1, shuffle=False, num_workers=0)
    
    for i in train_loader:
        print(i.shape)