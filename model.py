import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
        return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 3),
          nn.ReLU(inplace = 1),
          nn.Conv2d(in_channels, out_channels, 3),
          nn.ReLU(inplace = 1)
        )
    
class Unet(nn.Module):
    
    def __init__(self):
        super(Unet, self).__init__()
        
        self.conv1 = double_conv(1, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.maxpooling = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpooling(conv1)
        
        conv2 = self.conv2(x)
        x = self.maxpooling(conv2)
        
        conv3 = self.conv3(x)
        x = self.maxpooling(conv3)
        
        conv4 = self.conv4(x)
        x = self.maxpooling(conv4)
        
        conv5 = self.conv5(x)
        x = self.maxpooling(conv5)
        
        return x
        
model = Unet()
print(model)
