# uses padded convolutions (unlike the original paper, which uses valid convolutions)
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# each layer in decoder has 2 3x3conv+ReLU followed by maxpool

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # same convolution: input size = output size
            nn.BatchNorm2d(out_channels), # bias=False because batchnorm already has bias
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module): # modules needed for forward pass

    # paper has 2 output channels, this implementation is for binary segmentation
    # features is a list of number of features in each layer of the decoder/encoder
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]): 

        # initialize nn.Module
        super(UNet, self).__init__() 

        # ModuleList instead of normal list to register list elements as submodules of the module UNet
        self.downs = nn.ModuleList() # downsampling, encoder
        self.ups = nn.ModuleList() # upsampling, decoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        # maxpooling floors the division, this might cause problems when upsampling

        # encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, out_channels=feature))
            in_channels = feature # update in_channels for next layer

        # bottleneck
        self.bottleneck = DoubleConv(
            features[-1], # last element in features list
            features[-1]*2) # double the number of features at the end of the encoder

        # decoder
        # transpose convolution (they sometimes create artifacts) or bilinear followed by DoubleConv (can bypass artifacts)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d( # up-conv 2x2
                in_channels=feature*2, # x2 because skip connections 
                out_channels=feature, 
                kernel_size=2, 
                stride=2)) 
            self.ups.append(DoubleConv(feature*2, feature)) # double conv3x3,ReLU

        # final layer 1x1conv
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            # save output of each layer for skip connections before pooling
            # first element has the highest resolution and least features
            skip_connections.append(x) 
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse list for correct order

        for idx in range(0, len(self.ups), 2): # iterate by 2 because there is two elements in each layer
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # sizes might mismatch due to maxpooling/upconv (original paper uses cropping instead of resizing)
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along the channel dimension
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
    