import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = SelfAttention(1, channels)

    def forwward(self, x: torch.Tensor) -> torch.Tensor:
        #x : batch_size, features, height, width
        residual = x
        n, c, h, w = x.shape()

        #--> batch_size, features, height*width
        #each item represent a pixel (height * width)
        x = x.view((n, c, h*w))
        #--> batch_size, height*width, features
        #sequence of pixel, each fixel have features (just like token in transformer)
        x = x.transpose(-1, -2)

        x = self.attention(x)
        #--> batch_size, features, height*width
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residual
        return x



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: batch_size, in_channels, height, width 

        residual = x

        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)
    

class VAE_Decoder(nn.Sequential):
    def __init__():
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            #batch_size, 512, height/8, width/8
            VAE_ResidualBlock(512, 512),
            #batch_size, 512, height/4, width/4
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #batch_size, 512, height/2, width/2
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            #batch_size, 256, height, width
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            #batch_size, 3, height, width
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: batch_size, 4, height/8, width/8
        x /= 0.18215

        for module in self:
            x = module(x)
        #x: batch_size, 3, height, width
        return x