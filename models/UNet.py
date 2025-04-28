import torch
import torch.nn as nn

class Double_Convolution(nn.Module): #standard building block in U-Net: 2× Conv + BatchNorm + ReLU.
    def __init__(self, in_channels, out_channels):
        super(Double_Convolution, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )

    def forward (self, x):
        return self.conv(x)
    

class Encoder_Block(nn.Module): #Encoder block of U-Net: 2× Conv + MaxPool
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()
        self.conv=(Double_Convolution(in_channels, out_channels))
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x=self.conv(x)
        p=self.pool(x)
        return x,p

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()
        self.up= nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv=Double_Convolution(in_channels, out_channels)

    def forward(self,inputs, skip_connection):
        x=self.up(inputs)
        x=torch.cat((x, skip_connection), dim=1)
        x=self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        #Encoder 

        self.e1= Encoder_Block(in_channels, features[0])
        self.e2= Encoder_Block(features[0], features[1])
        self.e3= Encoder_Block(features[1], features[2])
        self.e4= Encoder_Block(features[2], features[3]) #Passing it through the layers
        
        #Bottleneck
        self.bottleneck=Double_Convolution(features[3], features[3]*2)

        #Decoder
        self.d1= Decoder_Block(features[3]*2, features[3])
        self.d2= Decoder_Block(features[3], features[2])
        self.d3= Decoder_Block(features[2], features[1])
        self.d4= Decoder_Block(features[1], features[0])

        #Final classifier layer

        self.outputs= nn.Conv2d(features[0], out_channels, kernel_size=1)
        