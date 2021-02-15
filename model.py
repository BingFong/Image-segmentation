import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch
from torchvision import transforms

from torch.utils.data import DataLoader

def double_conv(in_c, out_c):
    '''repeat conv and Relu twice'''
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size = 3, padding=1),
        nn.ReLU(inplace = 1),
        nn.Conv2d(out_c, out_c, kernel_size = 3, padding=1),   
        nn.ReLU(inplace = 1)
        )
    
class Unet(nn.Module):
    
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.maxpooling = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        conv1 = self.conv1(x) #w = h = 256
        x = self.maxpooling(conv1) #w, h = 128
        
        conv2 = self.conv2(x)
        x = self.maxpooling(conv2) #w = h = 64
        
        conv3 = self.conv3(x)
        x = self.maxpooling(conv3) #w = h = 32
        
        conv4 = self.conv4(x) #w = h = 32
        x = self.maxpooling(conv4) #w = h = 16

        conv5 = self.conv5(x) #w = h = 16
               
        x = self.upsample(conv5) #w = h = 32
    
        x = torch.cat((conv4, x), 1) #w = h = 32
        
        return x



    
from PIL import Image
img = Image.open(r'D:\python\Unet\project\Image-segmentation\test.tif').convert('RGB') #w, h = 256

m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
           
compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s)
    ])

img = compose(img)
input_batch = img.unsqueeze(0)
#data = DataLoader(img, batch_size=1)
#print(type(img1))

model = Unet()
if torch.cuda.is_available():
    device = torch.device("cuda")
    input_batch = input_batch.to(device)
    model.to(device)
    
a = model.forward(input_batch)
# print(a)
