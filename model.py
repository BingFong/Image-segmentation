import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image

import dataset

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
        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024) #size=16
        
        self.up4 = double_conv(512+1024, 512)
        self.up3 = double_conv(256+512, 256)
        self.up2 = double_conv(128+256, 128)
        self.up1 = double_conv(64+128, 64)
        
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.last_conv =  nn.Conv2d(64, 1, kernel_size = 1)
        
    def forward(self, x):
        conv1 = self.down1(x) #size=256
        x = self.maxpool(conv1) #size=128
        
        conv2 = self.down2(x) #size=128
        x = self.maxpool(conv2) #size=64
        
        conv3 = self.down3(x) #size=64
        x = self.maxpool(conv3) #size=32
        
        conv4 = self.down4(x) #size=32
        x = self.maxpool(conv4) #size=16

        conv5 = self.conv5(x) #size=16
        
        x = self.upsample(conv5) #size=32
        x = torch.cat((conv4, x), 1) #size=32, channels=512+1024
        up_conv4 = self.up4(x) #size=32, channels=512
        
        x = self.upsample(up_conv4) #size=64, channels=512
        x = torch.cat((conv3, x), 1) #size=64, channels=512+256
        up_conv3 = self.up3(x) #size=64, channels=256
        
        x = self.upsample(up_conv3) #size=128, channels=256
        x = torch.cat((conv2, x), 1) #size=128, channels=128+256
        up_conv2 = self.up2(x) #size=128, channels=128
        
        
        x = self.upsample(up_conv2) #size=256, channels=128
        x = torch.cat((conv1, x), 1) #size=256, channels=64+128
        up_conv1 = self.up1(x) #size=256, channels=64
    
        out = self.last_conv(up_conv1)
        out = torch.sigmoid(out)
        
        return out
    
    

if __name__ == '__main__': 
    
    # img = Image.open(r'D:\python\Unet\project\Image-segmentation\test.tif').convert('RGB') #w, h = 256
    
    # m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
               
    # compose = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=m, std=s)
    #     ])
    
    # img = compose(img)
    # input_batch = img.unsqueeze(0)
    data = dataset.get_data()
    model = Unet()
    #print('total params:', sum(p.numel() for p in model.parameters()))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        data = data.to(device)
        model.to(device)
        
    x = model.forward(data)
